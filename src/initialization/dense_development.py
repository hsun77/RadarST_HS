from __future__ import annotations
from pathlib import Path
import concurrent.futures,json,math,time
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from . import core
from .pipeline import PROJECT,manifests,path,context,truth,base,sha,dump,seal,trusted,guard,summarize


def dense_pool(run,sample):
    from .legacy import greedy
    f=core.functions();ctx=context(sample);state=np.load(path(run,sample.dataset)/'preprocess/state.npz')
    sparse=pd.read_parquet(path(run,sample.dataset)/'candidates/pool.parquet')
    rows=sparse.to_dict('records');seen=set(sparse.member_spots);scores=state['cfar'];graph=f.internal.adjacency(ctx['coords'])
    for seed in range(len(ctx['spot_ids'])):
        for step in greedy.generate_full_path(seed,scores,graph,ctx['spot_ids']):
            n=step['prefix_size'];members=step['member_spots']
            if n not in [6,7,9] or members in seen:continue
            seen.add(members);typ=f'core_{n}';cid=core.candidate_identity(sample.dataset,typ,members.split(';'))
            rows.append(dict(dataset_id=sample.dataset,sample_id=sample.sample,candidate_id=cid,candidate_row_key=cid,candidate_key=sample.dataset+'||'+typ+'||'+members,proposal_type=typ,member_spots=members,canonical_member_spots=members,n_spots=n,proposal_score=step['formal_path_proposal_score'],proposal_params=f'seed={seed};target={n}'))
    frame=pd.DataFrame(rows)
    # Re-rank the complete new pool, rather than importing sparse feature arrays.
    grouped=frame.groupby(['dataset_id','proposal_type'],sort=False)
    frame['score_per_spot']=frame.proposal_score/np.sqrt(frame.n_spots)
    frame['proposal_percentile']=grouped.proposal_score.transform(f.i.percentile_rank)
    frame['score_per_spot_percentile']=grouped.score_per_spot.transform(f.i.percentile_rank)
    frame['compactness_raw']=1/np.log1p(1+frame.n_spots)
    frame['compactness_percentile']=grouped.compactness_raw.transform(f.i.percentile_rank)
    return f.i.add_multiscale_persistence(frame)


def feature_task(task):
    root,d,m=task;run=Path(root);guard(run,'dense_selector')
    with threadpool_limits(limits=1):
        samples,pairs=manifests();sample=next(s for s in samples.itertuples() if s.dataset==d);row=next(r for r in pairs.itertuples() if r.dataset==d and r.method==m)
        work=run/'experiments/dense';poolpath=work/'samples'/d/'pool.parquet';folder=work/'pairs'/d/m;folder.mkdir(parents=True,exist_ok=True)
        inp={'pool':sha(poolpath),'state':sha(path(run,d)/'preprocess/state.npz'),'upstream':sha(PROJECT/row.upstream),'truth':sha(PROJECT/'data/annotations'/f'{d}.csv'),'sparse_teacher_targets':sha(path(run,d,m)/'targets/targets.parquet')}
        if trusted(folder,inp):return {'dataset':d,'method':m,'status':'CURRENT_RUN'}
        ctx=context(sample);frame=pd.read_parquet(poolpath);state=np.load(path(run,d)/'preprocess/state.npz')
        expr=pd.read_parquet(work/'samples'/d/'expression.parquet')
        frame=core.features(frame,base(row,ctx),ctx,state,m,expr);frame.to_parquet(folder/'features.parquet',index=False)
        components=core.components(truth(sample),ctx,sample.source);frame=core.utility_targets(frame,base(row,ctx),ctx,components)
        sparse_teacher=pd.read_parquet(path(run,d,m)/'targets/targets.parquet')
        positive=set(sparse_teacher.loc[sparse_teacher.teacher_E7_selected.eq(1),'candidate_id'])
        frame['teacher_E7_selected']=frame.candidate_id.isin(positive).astype(int)
        core.functions().g.sample_frame(frame).to_parquet(folder/'gain.parquet',index=False);seal(folder,inp)
        return {'dataset':d,'method':m,'status':'FRESH'}


def execute(run):
    from .legacy import dense
    f=core.functions();samples,pairs=manifests();dev=samples[samples.cohort.eq('development')];work=run/'experiments/dense';work.mkdir(parents=True,exist_ok=True)
    dump(work/'PROTOCOL.json',{'variants':['S0_FORMAL','S1_DENSE_OLD_QUOTA','S2_DENSE_RELAXED','S3_SPARSE_RELAXED'],'sizes':list(range(1,11)),'models_retrained':True,'external_additional_variants_not_authorized_without_gate':True,'seed':20260811,'formal_score':[.80,.20,-.025],'retention':.97})
    for s in dev.itertuples():
        folder=work/'samples'/s.dataset;folder.mkdir(parents=True,exist_ok=True)
        inp={'sparse_pool':sha(path(run,s.dataset)/'candidates/pool.parquet'),'state':sha(path(run,s.dataset)/'preprocess/state.npz')}
        if not trusted(folder,inp):
            frame=dense_pool(run,s);frame.to_parquet(folder/'pool.parquet',index=False)
            core.expression_features(frame,context(s),np.load(path(run,s.dataset)/'preprocess/state.npz')).to_parquet(folder/'expression.parquet',index=False);seal(folder,inp)
        print('DENSE_POOL',s.dataset,flush=True)
    tasks=[(str(run),r.dataset,r.method) for r in pairs[pairs.cohort.eq('development')].itertuples()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(feature_task,tasks):print('DENSE_FEATURES',result,flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(train_task,[(str(run),d) for d in dev.dataset]):print('DENSE_TRAINED',result,flush=True)
    events=[];audits=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(infer_task,tasks):print('DENSE_INFERRED',result,flush=True)
    for _,d,m in tasks:
        folder=work/'pair_results'/d/m;events.append(pd.read_csv(folder/'events.csv.gz'));audits.append(pd.read_csv(folder/'audit.csv'))
    master=pd.concat(events,ignore_index=True);master.to_csv(work/'EVENTS.csv.gz',index=False,compression='gzip');pd.concat(audits,ignore_index=True).to_csv(work/'SELECTION_AUDIT.csv',index=False)
    rows=[]
    for variant,g in master.groupby('variant'):
        result,_,_=summarize(g,seed=20260812);rows.append({'variant':variant,**result})
    pd.DataFrame(rows).to_csv(work/'SUMMARY.csv',index=False)
    dump(work/'STATUS.json',{'status':'FRESH','development_pairs':168,'variants':4,'external_additional_variants':'NOT RUN; require predefined development gate review','historical_cache_used':False})


def train_task(task):
    root,d=task;run=Path(root);guard(run,'dense_selector');work=run/'experiments/dense';samples,pairs=manifests();folder=work/'models'/d;inp={};parts=[];registry=[]
    with threadpool_limits(limits=1):
        for r in pairs[pairs.cohort.eq('development')&pairs.dataset.ne(d)].sort_values(['dataset','method']).itertuples():
            p=work/'pairs'/r.dataset/r.method/'gain.parquet';digest=sha(p);inp[str(p.relative_to(work))]=digest;parts.append(pd.read_parquet(p));registry.append({'path':str(p.relative_to(run)),'dataset':r.dataset,'method':r.method,'sha256':digest})
        if trusted(folder,inp):return {'dataset':d,'status':'CURRENT_RUN'}
        folder.mkdir(parents=True,exist_ok=True);train=pd.concat(parts,ignore_index=True);assert not train.dataset_id.eq(d).any();types=sorted(train.proposal_type.unique());f=core.functions()
        model=f.k.fit_models(train,types);model['proposal_types']=types;model['training_datasets']=sorted(set(train.dataset_id));model['held_out']=d
        joblib.dump(model,folder/'expected_gain.joblib',compress=3);pd.DataFrame(registry).to_csv(folder/'TRAINING_INPUTS.csv',index=False);dump(folder/'TRAINING_AUDIT.json',{'held_out':d,'training_datasets':model['training_datasets'],'rows':len(train),'external_rows':0,'held_out_rows':0,'hyperparameter_search':False,'feature_columns':model['feature_columns'],'configuration_source':'Unchanged source expected-gain hyperparameters; expanded-pool proposal one-hot columns'});seal(folder,inp)
    return {'dataset':d,'status':'FRESH'}


def infer_task(task):
    root,d,m=task;run=Path(root);guard(run,'dense_selector');samples,pairs=manifests();sample=next(s for s in samples.itertuples() if s.dataset==d);row=next(r for r in pairs.itertuples() if r.dataset==d and r.method==m)
    work=run/'experiments/dense';out=work/'pair_results'/d/m;out.mkdir(parents=True,exist_ok=True)
    inp={'dense_model':sha(work/'models'/d/'expected_gain.joblib'),'dense_features':sha(work/'pairs'/d/m/'features.parquet'),'slots':sha(path(run,d,m)/'inference/selector_slots.csv'),'sparse_ranking':sha(path(run,d,m)/'inference/ranking.parquet')}
    if trusted(out,inp):return {'dataset':d,'method':m,'status':'CURRENT_RUN'}
    f=core.functions();from .legacy import dense
    cfg=json.loads((PROJECT/'configs/formal.json').read_text());events=[];audits=[]
    with threadpool_limits(limits=1):infer_subset(run,work,samples[samples.dataset.eq(d)],pairs[pairs.dataset.eq(d)&pairs.method.eq(m)],f,dense,events,audits)
    pd.concat(events,ignore_index=True).to_csv(out/'events.csv.gz',index=False,compression='gzip');pd.DataFrame(audits).to_csv(out/'audit.csv',index=False);seal(out,inp)
    return {'dataset':d,'method':m,'status':'FRESH'}


def infer_subset(run,work,dev,pairs,f,dense,events,audits):
    for s in dev.itertuples():
        ctx=context(s);comps=core.components(truth(s),ctx,s.source);model=joblib.load(work/'models'/s.dataset/'expected_gain.joblib')
        for r in pairs[pairs.dataset.eq(s.dataset)].itertuples():
            sparse=pd.read_parquet(path(run,r.dataset,r.method)/'features/features.parquet');expanded=pd.read_parquet(work/'pairs'/r.dataset/r.method/'features.parquet')
            slots=pd.read_csv(path(run,r.dataset,r.method)/'inference/selector_slots.csv');original=base(r,ctx)
            for variant in ['S0_FORMAL','S1_DENSE_OLD_QUOTA','S3_SPARSE_RELAXED','S2_DENSE_RELAXED']:
                dense_variant=variant in ['S1_DENSE_OLD_QUOTA','S2_DENSE_RELAXED'];frame=expanded if dense_variant else sparse
                ranking=f.k.predict(frame,model,model['proposal_types']) if dense_variant else pd.read_parquet(path(run,r.dataset,r.method)/'inference/ranking.parquet')
                score=.80*frame.score_per_spot_percentile.to_numpy()+.20*ranking.score_expected_gain_rank.to_numpy()-.025*frame.cross_component_risk.to_numpy()
                if variant in ['S2_DENSE_RELAXED','S3_SPARSE_RELAXED']:
                    selected,search=dense.relaxed_assignment(f.sel,frame,score,slots,ctx['spot_ids'],r.method,variant)
                else:
                    cfg=json.loads((PROJECT/'configs/formal.json').read_text());_,selected,search=f.sel.exact_token_assignment(frame,score,score,slots,ctx['spot_ids'],r.method,cfg['selector_variants']['development'])
                total=int(slots.n_spots.sum());assign,retained,burden,target=dense.retained_selection(selected,total,r.method,variant)
                labels=original.copy()
                for spot,label in assign.items():labels.loc[spot]=label
                folder=work/'predictions'/variant/r.dataset/r.method;folder.mkdir(parents=True,exist_ok=True)
                ranking=ranking.copy();ranking['candidate_id']=frame.candidate_id.to_numpy();ranking['candidate_key']=frame.candidate_key.to_numpy();ranking.to_parquet(folder/'ranking.parquet',index=False)
                retained.to_csv(folder/'regions.csv',index=False);pd.DataFrame({'spot_id':ctx['spot_ids'],'pred_label':labels.to_numpy()}).to_csv(folder/'labels.csv',index=False)
                ev,macro=core.evaluate(original,labels,truth(s),ctx,comps,r.method,'development');ev['variant']=variant;events.append(ev)
                audits.append({'dataset':r.dataset,'method':r.method,'variant':variant,'modified_spots':burden,'target_retained_spots':target,'difference':abs(burden-target),'formal_slot_spots':total,**{k:v for k,v in search.items() if k!='attempt_diagnostics'}})
            print('DENSE_INFERRED',r.dataset,r.method,flush=True)
