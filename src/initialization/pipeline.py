from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,os,sys,time,traceback,platform,subprocess,importlib.metadata,concurrent.futures
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import core

PROJECT=Path(__file__).resolve().parents[2]
PRIMARY_RUN='clean_v3_source_cohort_rng'
MODEL_CACHE={}
STAGES=['validate','preprocess','generate_candidates','features','targets','train_loso','train_final','development','external','controls','cfar_biology','proposal_audit','dense_selector','tables','figures','verify']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,default=str),encoding='utf-8')

def training_signature():
    files=['t','signed','k','g','d','targets','a']
    value={'fit_sources':{name:sha(PROJECT/'src/radarst/legacy'/f'{name}.py') for name in files},'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','joblib']},'identity':'full SHA256 canonical membership; target startup hashseed=0'}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def verify_release_source_lock():
    lock=PROJECT/'configs/SOURCE_LOCK.json'
    if not lock.exists():return
    source=json.loads(lock.read_text())
    for rel,digest in source['files'].items():
        p=PROJECT/rel
        if not p.exists() or sha(p)!=digest:
            raise RuntimeError('RELEASE_SOURCE_CHANGED: '+rel+'. Do not resume sealed outputs under changed code; create a reviewed new release and empty run.')
def manifests():return pd.read_csv(PROJECT/'manifests/SAMPLES.csv',keep_default_na=False),pd.read_csv(PROJECT/'manifests/PAIRS.csv',keep_default_na=False)
def context(sample):
    coords=pd.read_csv(PROJECT/'data/coordinates'/f'{sample.dataset}.csv',dtype={'spot_id':str})
    if sample.source=='external_second':
        geometry=pd.read_csv(PROJECT/sample.metadata,usecols=['spot_id','array_col','array_row'],dtype={'spot_id':str})
        coords=coords.merge(geometry,on='spot_id',how='left',validate='one_to_one')
        if coords[['array_col','array_row']].isna().any().any():raise RuntimeError('HER2ST truth coordinate alignment missing')
    return core.make_context(sample.dataset,sample.sample,coords,sample.source)
def truth(sample):
    df=pd.read_csv(PROJECT/'data/annotations'/f'{sample.dataset}.csv',dtype=str,keep_default_na=False)
    return pd.Series(df.truth_label.to_numpy(),index=df.spot_id)
def base(row,ctx):return core.read_upstream(PROJECT/row.upstream,row.barcode_col,row.label_col,ctx['spot_ids'])
def path(run,d,m=None):return run/'pairs'/d/m if m is not None else run/'samples'/d
def seal(folder,inputs):
    files=[{'path':str(p.relative_to(folder)),'sha256':sha(p)} for p in folder.rglob('*') if p.is_file() and p.name!='SEAL.json']
    dump(folder/'SEAL.json',{'inputs':inputs,'files':files,'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
def trusted(folder,inputs):
    p=folder/'SEAL.json'
    if not p.exists():return False
    d=json.loads(p.read_text())
    return d['inputs']==inputs and all((folder/x['path']).exists() and sha(folder/x['path'])==x['sha256'] for x in d['files'])

def validate(run):
    samples,pairs=manifests();audit=[];failed=[]
    declared_truth_fields=json.loads((PROJECT/'configs/annotation_policy.json').read_text())['development_truth_fields']
    inputs=pd.read_csv(PROJECT/'manifests/INPUT_SHA256.csv')
    for row in inputs.itertuples():
        if sha(PROJECT/row.path)!=row.sha256:raise ValueError('Imported input changed: '+row.path)
    for sample in samples.itertuples():
        meta=pd.read_csv(PROJECT/sample.metadata,sep='\t' if sample.metadata.endswith('.tsv') else ',',keep_default_na=False,dtype=str)
        bc=next((c for c in ['spot_id','ID','barcode','spot_id_normalized','Unnamed: 0'] if c in meta),None)
        if bc is None:raise ValueError('No explicit metadata barcode '+sample.dataset)
        xcol=next((c for c in ['pxl_col_in_fullres','imagecol','x','array_col','col'] if c in meta),None)
        ycol=next((c for c in ['pxl_row_in_fullres','imagerow','y','array_row','row'] if c in meta),None)
        if sample.dataset in declared_truth_fields:tcol=declared_truth_fields[sample.dataset]
        elif sample.cohort=='development':raise ValueError('Development truth field must be explicitly registered: '+sample.dataset)
        elif sample.source=='external_second':tcol='truth_label_normalized'
        elif sample.source=='external_first':tcol='ground_truth'
        else:tcol=next((c for c in ['layer_guess_reordered_short','layer_guess_reordered','Layer','ground_truth'] if c in meta),None)
        if tcol not in meta:raise ValueError('Source-declared truth field missing: '+sample.dataset+' '+str(tcol))
        if any(v is None for v in [xcol,ycol,tcol]):raise ValueError('Metadata schema unresolved '+sample.dataset)
        coords=pd.DataFrame({'spot_id':meta[bc].astype(str),'x':pd.to_numeric(meta[xcol],errors='coerce'),'y':pd.to_numeric(meta[ycol],errors='coerce')})
        if coords.spot_id.duplicated().any():raise ValueError('Duplicate metadata barcode '+sample.dataset)
        valid=coords.x.notna()&coords.y.notna()
        data=core.read_expression(PROJECT/sample.cfar_expression,'10x_h5');valid &= coords.spot_id.isin(data.obs_names)
        coords=coords.loc[valid].reset_index(drop=True)
        ann=pd.DataFrame({'spot_id':meta.loc[valid,bc].astype(str).to_numpy(),'truth_label':meta.loc[valid,tcol].astype(str).to_numpy()})
        if sample.dataset=='A01':ann.loc[ann.truth_label.isin(['Uncertain','Uncertain_overlap']),'truth_label']='UNASSIGNED'
        if sample.source=='external_second' and 'truth_label_role' in meta:
            ann.loc[meta.loc[valid,'truth_label_role'].to_numpy()=='excluded_source_undetermined','truth_label']='UNASSIGNED'
        coordsdir=PROJECT/'data/coordinates';anndir=PROJECT/'data/annotations';coordsdir.mkdir(exist_ok=True);anndir.mkdir(exist_ok=True)
        coords.to_csv(coordsdir/f'{sample.dataset}.csv',index=False);ann.to_csv(anndir/f'{sample.dataset}.csv',index=False)
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            try:
                labels=core.read_upstream(PROJECT/row.upstream,row.barcode_col,row.label_col,coords.spot_id.tolist())
                status='PASS';error=''
            except Exception as e:status='BLOCKED';error=str(e);failed.append({'dataset':row.dataset,'method':row.method,'error':error})
            audit.append({'dataset':row.dataset,'cohort':sample.cohort,'method':row.method,'source_truth_column':tcol,'n_metadata':len(meta),'n_expression':data.n_obs,'n_coordinates':len(coords),'dropped_invalid_or_no_expression':int((~valid).sum()),'status':status,'error':error,'missing_pred_sentinel_spots':int(labels.eq('missing_pred').sum()) if status=='PASS' else None,'n_pred_clusters':int(labels.nunique()) if status=='PASS' else None,'positional_fallback':False})
        print('VALIDATED',sample.dataset,len(coords),flush=True)
    pd.DataFrame(audit).to_csv(run/'INPUT_VALIDATION.csv',index=False);dump(run/'INPUT_ERRORS.json',failed)
    blocked_dev=[r for r in failed if r['dataset'] in set(samples[samples.cohort.eq('development')].dataset)]
    if blocked_dev:raise RuntimeError('Development alignment failures: '+repr(blocked_dev))

def preprocess(run):
    samples,_=manifests()
    for sample in samples[samples.cohort.ne('control')].itertuples():
        folder=path(run,sample.dataset)/'preprocess';inp={k:sha(PROJECT/getattr(sample,k)) for k in ['cfar_expression','feature_expression']};inp['coords']=sha(PROJECT/'data/coordinates'/f'{sample.dataset}.csv')
        if trusted(folder,inp):continue
        folder.mkdir(parents=True,exist_ok=True);t=time.time()
        state=core.preprocess(sample,PROJECT);np.savez_compressed(folder/'state.npz',**state);seal(folder,inp)
        print('PREPROCESSED',sample.dataset,'seconds',time.time()-t,flush=True)

def generate_candidates(run):
    samples,_=manifests()
    for sample in samples[samples.cohort.ne('control')].itertuples():
        folder=path(run,sample.dataset)/'candidates';raw=path(run,sample.dataset)/'preprocess/state.npz';inp={'preprocess':sha(raw),'config':sha(PROJECT/'configs/formal.json')}
        if trusted(folder,inp):continue
        folder.mkdir(parents=True,exist_ok=True);coords=pd.read_csv(PROJECT/'data/coordinates'/f'{sample.dataset}.csv',dtype={'spot_id':str})
        frame=core.generate(sample,coords,np.load(raw));frame.to_parquet(folder/'pool.parquet',index=False);seal(folder,inp)
        print('CANDIDATES',sample.dataset,len(frame),flush=True)

def features(run):
    samples,pairs=manifests()
    tasks=[]
    for sample in samples[samples.cohort.ne('control')].itertuples():
        ctx=context(sample);pool=path(run,sample.dataset)/'candidates/pool.parquet';statepath=path(run,sample.dataset)/'preprocess/state.npz';frame=pd.read_parquet(pool);state=np.load(statepath)
        exprdir=path(run,sample.dataset)/'expression_features';exprinputs={'pool':sha(pool),'state':sha(statepath)}
        if not trusted(exprdir,exprinputs):
            exprdir.mkdir(parents=True,exist_ok=True)
            core.expression_features(frame,ctx,state).to_parquet(exprdir/'features.parquet',index=False);seal(exprdir,exprinputs)
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            folder=path(run,row.dataset,row.method)/'features';inp={'pool':sha(pool),'state':sha(statepath),'upstream':sha(PROJECT/row.upstream)}
            if trusted(folder,inp):continue
            tasks.append((str(run),sample.dataset,row.method))
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(feature_worker,tasks):
            if result.get('error'):errors.append(result)
            else:print('FEATURES',result['dataset'],result['method'],result['rows'],'seconds',result['seconds'],flush=True)
    if errors:
        dump(run/'logs/feature_errors.json',errors);raise RuntimeError('Feature workers failed: '+repr(errors[:3]))

def feature_worker(task):
    runstr,d,m=task;run=Path(runstr);globals()['CURRENT_STAGE']='features';guard(run,'features')
    try:
        with threadpool_limits(limits=1):
            samples,pairs=manifests();sample=next(r for r in samples.itertuples() if r.dataset==d);row=next(r for r in pairs.itertuples() if r.dataset==d and r.method==m)
            ctx=context(sample);pool=path(run,d)/'candidates/pool.parquet';statepath=path(run,d)/'preprocess/state.npz';frame=pd.read_parquet(pool);state=np.load(statepath)
            inp={'pool':sha(pool),'state':sha(statepath),'upstream':sha(PROJECT/row.upstream)};folder=path(run,d,m)/'features';folder.mkdir(parents=True,exist_ok=True);t=time.time()
            expression=pd.read_parquet(path(run,d)/'expression_features/features.parquet')
            out=core.features(frame,base(row,ctx),ctx,state,m,expression)
            out.to_parquet(folder/'features.parquet',index=False);core.functions().g.feature_frame(out,sorted(frame.proposal_type.unique())).to_parquet(folder/'104_features.parquet',index=False);seal(folder,inp)
            return {'dataset':d,'method':m,'rows':len(out),'seconds':time.time()-t}
    except Exception:return {'dataset':d,'method':m,'error':traceback.format_exc()}

def targets(run):
    samples,pairs=manifests();config=json.loads((PROJECT/'configs/teacher_protocol.json').read_text());f=core.functions()
    for sample in samples[samples.cohort.eq('development')].itertuples():
        ctx=context(sample);pool=pd.read_parquet(path(run,sample.dataset)/'candidates/pool.parquet');ps=pairs[pairs.dataset.eq(sample.dataset)]
        bases={row.method:base(row,ctx) for row in ps.itertuples()};keys,audit=core.teacher_targets(pool,ctx,bases,config)
        comps=core.components(truth(sample),ctx,sample.source);audit.to_csv(path(run,sample.dataset)/'teacher_generation.csv',index=False)
        comps.assign(truth_spots=comps.truth_spots.map(';'.join)).to_csv(path(run,sample.dataset)/'truth_components.csv',index=False)
        for row in ps.itertuples():
            featurepath=path(run,row.dataset,row.method)/'features/features.parquet';folder=path(run,row.dataset,row.method)/'targets';inp={'features':sha(featurepath),'truth':sha(PROJECT/'data/annotations'/f'{sample.dataset}.csv'),'teacher_config':sha(PROJECT/'configs/teacher_protocol.json'),'python_hash_seed':os.environ.get('PYTHONHASHSEED','UNRECORDED'),'target_source':sha(PROJECT/'src/radarst/legacy/targets.py')}
            if trusted(folder,inp):continue
            folder.mkdir(parents=True,exist_ok=True);frame=pd.read_parquet(featurepath);frame['teacher_E7_selected']=frame.candidate_id.isin(keys[row.method]).astype(int)
            frame=core.utility_targets(frame,bases[row.method],ctx,comps)
            f.d.deterministic_training_sample(frame).to_parquet(folder/'teacher.parquet',index=False)
            f.targets.training_sample(frame).to_parquet(folder/'signed.parquet',index=False)
            f.g.sample_frame(frame).to_parquet(folder/'gain.parquet',index=False)
            frame[['candidate_id','candidate_key','teacher_E7_selected','candidate_truth_best_IoU','candidate_truth_component_id','base_best_IoU_for_candidate_target','signed_utility_target','utility_target']].to_parquet(folder/'targets.parquet',index=False)
            seal(folder,inp);print('TARGETS',row.dataset,row.method,'teacher',int(frame.teacher_E7_selected.sum()),'positive',int(frame.signed_utility_target.gt(0).sum()),flush=True)

def train_fold(run,heldout):
    samples,pairs=manifests();datasets=sorted(samples[samples.cohort.eq('development')].dataset);training=[d for d in datasets if d!=heldout];f=core.functions()
    folder=run/'models'/('full_fit' if heldout=='FULL' else heldout);inp={'training_protocol':training_signature()};parts={'teacher':[],'signed':[],'gain':[]};registry=[]
    for d in training:
        for row in pairs[pairs.dataset.eq(d)].sort_values('method').itertuples():
            for name in parts:
                p=path(run,d,row.method)/'targets'/(name+'.parquet');h=sha(p);inp[str(p.relative_to(run))]=h;frame=pd.read_parquet(p)
                assert frame.dataset_id.eq(d).all() and heldout not in set(frame.dataset_id)
                parts[name].append(frame);registry.append({'path':str(p.relative_to(run)),'sha256':h,'dataset':d,'kind':name})
    if trusted(folder,inp):return
    folder.mkdir(parents=True,exist_ok=True);tables={k:pd.concat(v,ignore_index=True) for k,v in parts.items()};types=sorted(tables['teacher'].proposal_type.unique())
    teacher=f.t.fit_model(tables['teacher'],types);print('TEACHER_FIT',heldout,flush=True)
    classifier,regressor=f.signed.fit_signed_heads(tables['signed'],types);print('SIGNED_FIT',heldout,flush=True)
    ranker=f.k.fit_models(tables['gain'],types);print('GAIN_FIT',heldout,flush=True)
    bundle={'teacher':teacher,'signed_utility_classifier':classifier,'signed_utility_regressor':regressor,'proposal_types':types,'harm_weight':.5,'extra_utility_budget':.01,'training_datasets':training,'held_out':heldout,'configuration_selection_is_not_nested':True}
    joblib.dump(bundle,folder/'selector_heads.joblib',compress=3);joblib.dump(ranker,folder/'expected_gain.joblib',compress=3)
    pd.DataFrame(registry).to_csv(folder/'TRAINING_INPUTS.csv',index=False)
    dump(folder/'TRAINING_AUDIT.json',{'held_out':heldout,'training_datasets':training,'external_training_rows':0,'heldout_training_rows':0,'sampled_rows':{k:len(v) for k,v in tables.items()},'feature_columns':ranker['feature_columns'],'models':{k:bundle[k].get_params() for k in ['teacher','signed_utility_classifier','signed_utility_regressor']},'hyperparameter_search':False})
    seal(folder,inp)

def train_loso(run):
    samples,_=manifests()
    tasks=[(str(run),d) for d in sorted(samples[samples.cohort.eq('development')].dataset)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(train_worker,tasks):print('LOSO_COMPLETED',result,flush=True)

def train_worker(task):
    root,d=task;run=Path(root);guard(run,'train_loso')
    with threadpool_limits(limits=1):train_fold(run,d)
    return d
def train_final(run):train_fold(run,'FULL')

def verified_models(run,modeldir,heldout):
    sealpath=modeldir/'SEAL.json'
    if not sealpath.exists():raise RuntimeError('Unsealed model folder')
    key=(str(modeldir),sha(sealpath))
    if key in MODEL_CACHE:return MODEL_CACHE[key]
    registry=pd.read_csv(modeldir/'TRAINING_INPUTS.csv')
    for row in registry.itertuples():
        if sha(run/row.path)!=row.sha256:raise RuntimeError('Model targets are stale: '+str(modeldir)+' '+row.path)
        if heldout!='FULL' and row.dataset==heldout:raise RuntimeError('Held-out training input detected')
    if not (modeldir/'SEAL.json').exists():raise RuntimeError('Unsealed model folder')
    stored=json.loads((modeldir/'SEAL.json').read_text())
    if not trusted(modeldir,stored['inputs']):raise RuntimeError('Model seal verification failed')
    value=(joblib.load(modeldir/'selector_heads.joblib'),joblib.load(modeldir/'expected_gain.joblib'))
    MODEL_CACHE[key]=value
    return value

def infer_cohort(run,cohort,comparator=False):
    samples,pairs=manifests()
    tasks=[(str(run),r.dataset,r.method,cohort,comparator) for r in pairs[pairs.cohort.eq(cohort)].itertuples()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inference_worker,tasks):print('INFERENCE_COMPLETE',result,flush=True)
    events=[];macros=[]
    for row in pairs[pairs.cohort.eq(cohort)].itertuples():
        folder=path(run,row.dataset,row.method)/('comparator' if comparator else 'inference')
        events.append(pd.read_csv(folder/'events.csv'));macros.append(pd.read_csv(folder/'macro.csv'))
    out=run/'evaluation';out.mkdir(exist_ok=True);suffix='_comparator' if comparator else ''
    pd.concat(events,ignore_index=True).to_csv(out/(cohort+suffix+'_events.csv.gz'),index=False,compression='gzip')
    pd.concat(macros,ignore_index=True).to_csv(out/(cohort+suffix+'_macro.csv'),index=False)

def inference_worker(task):
    root,d,m,cohort,comparator=task;run=Path(root);guard(run,'inference')
    with threadpool_limits(limits=1):
        samples,pairs=manifests();s=samples[samples.dataset.eq(d)];p=pairs[pairs.dataset.eq(d)&pairs.method.eq(m)]
        infer_subset(run,cohort,comparator,s,p)
    return {'dataset':d,'method':m,'cohort':cohort,'comparator':comparator}

def infer_subset(run,cohort,comparator=False,samples=None,pairs=None):
    if samples is None or pairs is None:samples,pairs=manifests()
    config=json.loads((PROJECT/'configs/formal.json').read_text());events=[];macros=[]
    for sample in samples[samples.cohort.eq(cohort)].itertuples():
        ctx=context(sample);modeldir=run/'models'/(sample.dataset if cohort=='development' else 'full_fit')
        bundle,ranker=verified_models(run,modeldir,sample.dataset if cohort=='development' else 'FULL')
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            featurepath=path(run,row.dataset,row.method)/'features/features.parquet';folder=path(run,row.dataset,row.method)/('comparator' if comparator else 'inference')
            inp={'features':sha(featurepath),'selector_model':sha(modeldir/'selector_heads.joblib'),'ranker_model':sha(modeldir/'expected_gain.joblib'),'config':sha(PROJECT/'configs/formal.json'),'annotation_policy':sha(PROJECT/'configs/annotation_policy.json')}
            if not trusted(folder,inp):
                folder.mkdir(parents=True,exist_ok=True);frame=pd.read_parquet(featurepath);original=base(row,ctx)
                slots,regions,labels,ranking,search=core.infer(frame,original,ctx,bundle,ranker,config,row.method,comparator)
                slots.to_csv(folder/'selector_slots.csv',index=False);regions.to_csv(folder/'selected_regions.csv',index=False);ranking.to_parquet(folder/'ranking.parquet',index=False)
                pd.DataFrame({'spot_id':ctx['spot_ids'],'pred_label':labels.to_numpy()}).to_csv(folder/'final_labels.csv',index=False);dump(folder/'SEARCH.json',search)
                # Truth is read only after prediction and region files are complete.
                ref=truth(sample);component=core.components(ref,ctx,sample.source)
                ev,macro=core.evaluate(original,labels,ref,ctx,component,row.method,cohort)
                ev.to_csv(folder/'events.csv',index=False);pd.DataFrame([macro]).to_csv(folder/'macro.csv',index=False);seal(folder,inp)
                print('INFERRED',cohort,row.dataset,row.method,len(regions),flush=True)
            events.append(pd.read_csv(folder/'events.csv'));macros.append(pd.read_csv(folder/'macro.csv'))

def summarize(events,seed=20260720,repetitions=10000):
    small=events[events.truth_size.between(1,10)].copy();pair=small.groupby(['dataset','method'])[['before','after','gain']].mean().reset_index()
    sample=pair.groupby('dataset')[['before','after','gain']].mean();rng=np.random.default_rng(seed);draw=rng.integers(0,len(sample),(repetitions,len(sample)))
    ci=np.quantile(sample.gain.to_numpy()[draw].mean(1),[.025,.975])
    return {'before':sample.before.mean(),'after':sample.after.mean(),'gain':sample.gain.mean(),'ci_low':ci[0],'ci_high':ci[1],'samples':len(sample),'pairs':len(pair),'positive_pairs':int(pair.gain.gt(1e-12).sum()),'zero_pairs':int(pair.gain.abs().le(1e-12).sum()),'negative_pairs':int(pair.gain.lt(-1e-12).sum()),'improved_events':int(small.gain.gt(1e-12).sum()),'unchanged_events':int(small.gain.abs().le(1e-12).sum()),'decreased_events':int(small.gain.lt(-1e-12).sum()),'loss_event_fraction':small.gain.lt(-1e-12).mean(),'bootstrap_repetitions':repetitions},pair,sample

def tables(run):
    from .release import build_tables
    build_tables(run)
    out=run/'tables';out.mkdir(exist_ok=True)
    for cohort in ['development','external']:
        for mode in ['','_comparator']:
            p=run/'evaluation'/(cohort+mode+'_events.csv.gz')
            if not p.exists():continue
            ev=pd.read_csv(p)
            ev[ev.truth_size.between(1,10)].groupby(['truth_size','dataset','method'])[['before','after','gain']].mean().groupby(['truth_size','dataset']).mean().groupby('truth_size').mean().to_csv(out/(cohort+mode+'_exact_size.csv'))
            if not mode:
                ev.to_csv(out/(cohort+'_component_events.csv.gz'),index=False,compression='gzip')

def controls(run):
    samples,pairs=manifests();rows=[];validation=pd.read_csv(run/'INPUT_VALIDATION.csv')
    for sample in samples[samples.cohort.eq('control')].itertuples():
        ctx=context(sample);ref=truth(sample)
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            valid=validation[validation.dataset.eq(row.dataset)&validation.method.eq(row.method)]
            if not valid.status.eq('PASS').all():continue
            original=base(row,ctx);metrics=core.functions().b67.strict_metrics(ref,original);rows.append({'dataset':row.dataset,'method':row.method,**metrics})
    folder=run/'evaluation';folder.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(folder/'dlpfc_macro.csv',index=False)
    infer_cohort(run,'development',True);infer_cohort(run,'external',True)
    from .experiments import run_controls
    run_controls(run)

def guard(run,stage):
    runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    if os.environ.get('WINDIR'):runtime.append(Path(os.environ['WINDIR']).resolve())
    accessfile=run/'logs'/f'access_{os.getpid()}_{stage}.jsonl'
    accessfile.parent.mkdir(parents=True,exist_ok=True)
    accessfd=os.open(accessfile,os.O_CREAT|os.O_APPEND|os.O_WRONLY)
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        raw=os.fsdecode(args[0]);p=Path(raw).absolute().resolve()
        if raw.lower()=='nul':return
        mode,flags=args[1:3];write=isinstance(mode,str) and any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if not p.is_relative_to(PROJECT) and not any(p.is_relative_to(q) for q in runtime):raise RuntimeError('UNDECLARED_INPUT_BLOCKED: '+str(p))
        if write and not (p.is_relative_to(run) or globals().get('CURRENT_STAGE')=='validate' and any(p.is_relative_to(PROJECT/'data'/q) for q in ['coordinates','annotations'])):raise RuntimeError('WRITE_OUTSIDE_RUN_BLOCKED: '+str(p))
        if p.is_relative_to(PROJECT/'runs') and not p.is_relative_to(run):raise RuntimeError('OTHER_RUN_STATE_BLOCKED: '+str(p))
        if p.is_relative_to(PROJECT) and p!=accessfile:
            os.write(accessfd,(json.dumps({'path':str(p.relative_to(PROJECT)),'write':bool(write),'stage':globals().get('CURRENT_STAGE',stage),'pid':os.getpid()})+'\n').encode('utf-8'))
    sys.addaudithook(hook)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES+['all'],required=True);p.add_argument('--run',default=PRIMARY_RUN);p.add_argument('--held-out');args=p.parse_args()
    if not args.run.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe run name')
    verify_release_source_lock()
    run=PROJECT/'runs'/args.run;run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    os.environ['NUMBA_CACHE_DIR']=str(run/'numba_cache');os.environ['MPLCONFIGDIR']=str(run/'mplconfig');sys.dont_write_bytecode=True
    config=json.loads((PROJECT/'configs/formal.json').read_text());stages=STAGES if args.stage=='all' else [args.stage]
    log=[];exitstatus=0
    guard(run,args.stage)
    dump(run/'logs'/('code_snapshot_'+args.stage+'.json'),{'command':sys.argv,'python':sys.version,'hash_seed':os.environ.get('PYTHONHASHSEED'),'files':{str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'src').rglob('*.py')},'configuration_sha256':sha(PROJECT/'configs/formal.json')})
    for stage in stages:
        globals()['CURRENT_STAGE']=stage
        t=time.time();status='PASS';error=''
        try:
            print('STAGE_START',stage,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
            with threadpool_limits(limits=config['blas_threads'],user_api='blas'),threadpool_limits(limits=config['openmp_threads'],user_api='openmp'):
                if stage=='train_loso' and args.held_out:train_fold(run,args.held_out)
                elif stage in ['development','external']:infer_cohort(run,stage)
                elif stage in ['cfar_biology','proposal_audit','dense_selector']:
                    from . import experiments
                    getattr(experiments,stage)(run)
                elif stage in ['figures','verify']:
                    from . import release
                    getattr(release,stage)(run)
                elif stage in globals() and callable(globals()[stage]):globals()[stage](run)
                else:raise NotImplementedError('Stage has not yet been ported: '+stage)
        except Exception:
            status='BLOCKED';error=traceback.format_exc();(run/'logs'/(stage+'_traceback.txt')).write_text(error,encoding='utf-8');print(error,flush=True);exitstatus=1
        activity={}
        if stage in ['train_loso','train_final'] and status=='PASS':
            groups=[run/'models'/'full_fit'] if stage=='train_final' else [run/'models'/args.held_out] if args.held_out else [run/'models'/d for d in manifests()[0].query("cohort == 'development'").dataset]
            fitted=sum(any(p.stat().st_mtime>=t for p in folder.glob('*.joblib')) for folder in groups)
            activity={'training_groups':len(groups),'groups_fitted_this_invocation':fitted,'verified_same_task_reused_groups':len(groups)-fitted}
            if fitted==0:status='VERIFIED_REUSE'
            elif fitted<len(groups):status='PASS_WITH_VERIFIED_REUSE'
        log.append({'stage':stage,'status':status,'start_unix':t,'end_unix':time.time(),'seconds':time.time()-t,'error':error,**activity})
        dump(run/'logs'/('stage_'+stage+'.json'),log[-1]);print('STAGE_END',stage,status,flush=True)
    dump(run/'EXECUTION.json',log)
    dump(run/'logs'/('invocation_'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'.json'),{'command':sys.argv,'exit_code':exitstatus,'stages':log})
    sys.exit(exitstatus)
if __name__=='__main__':main()
