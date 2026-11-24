from __future__ import annotations
from pathlib import Path
from collections import Counter
import hashlib,json,platform,sys,time,subprocess,os
import joblib,numpy as np,pandas as pd
from .pipeline import PROJECT,manifests,path,sha,dump,summarize,context,truth,base,verified_models,trusted
from . import core

TABLE_NAMES={1:'Dataset manifest',2:'Workflow provenance',3:'Input alignment and completion',4:'Development annotation provenance and components',5:'External annotation provenance and components',6:'Label role policy',7:'Algorithm configuration',8:'Component event master',9:'Macro compatibility and burden',10:'Training and isolation audit',11:'Model/runtime provenance',12:'External section results',13:'External leave-one-section-out',14:'Pathology context summary',15:'Pathology exact-size records',16:'Denominator registry',17:'Workflow and section summaries',18:'Molecular-score comparator',19:'Candidate randomization and confidence intervals',20:'Parameter sensitivity',21:'Proposal exact-size oracle',22:'Proposal saturation',23:'Proposal counts',24:'CFAR score/gene concordance',25:'CFAR pathway classes',26:'Matched connected random calibration'}


def write_md(p,text):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')


def build_tables(run):
    out=run/'tables';out.mkdir(exist_ok=True);reports=run/'reports';reports.mkdir(exist_ok=True);samples,pairs=manifests();registry=[]
    def table(n,frame,source,notes='',status='FRESH'):
        dest=out/f'S{n:02d}.csv';frame.to_csv(dest,index=False)
        registry.append({'table':f'S{n}','title':TABLE_NAMES[n],'status':status,'path':str(dest.relative_to(run)),'rows':len(frame),'source':source,'notes':notes,'sha256':sha(dest)})
    def fromfile(n,p,notes=''):
        if p.exists():table(n,pd.read_csv(p),str(p.relative_to(run)),notes)
        else:registry.append({'table':f'S{n}','title':TABLE_NAMES[n],'status':'BLOCKED','path':'','rows':0,'source':str(p.relative_to(run)),'notes':'Required computation/output missing','sha256':''})
    table(1,samples,'manifests/SAMPLES.csv','Public raw accession/license enrichment remains to be independently verified.')
    table(2,pd.DataFrame({'method':sorted(pairs.method.unique()),'upstream_clustering_recomputed':False,'upstream_runtime_version':'NOT VERIFIED','upstream_device':'NOT VERIFIED'}),'manifests/PAIRS.csv','Upstream execution environments cannot be inferred from labels.',status='BLOCKED')
    table(3,pd.read_csv(run/'INPUT_VALIDATION.csv'),'INPUT_VALIDATION.csv','490 current barcode/cluster-count checks are fresh, but the manuscript execution/device/upstream-fallback fields are not independently complete.',status='BLOCKED')
    components=[];roles=[]
    for s in samples[samples.cohort.ne('control')].itertuples():
        ctx=context(s);comp=core.components(truth(s),ctx,s.source);comp['cohort']=s.cohort;comp['truth_spots']=comp.truth_spots.map(';'.join);components.append(comp)
        ann=pd.read_csv(PROJECT/'data/annotations'/f'{s.dataset}.csv',keep_default_na=False);counts=ann.truth_label.value_counts()
        scored=set(comp.truth_label);ncomp=comp.groupby('truth_label').size()
        for label,n in counts.items():roles.append({'dataset':s.dataset,'cohort':s.cohort,'label':label,'spots':n,'scored':label in scored,'components':int(ncomp.get(label,0)),'source_policy':s.source,'truth_training_allowed':s.cohort=='development'})
    component=pd.concat(components,ignore_index=True);component.to_csv(out/'FRESH_TRUTH_COMPONENTS.csv.gz',index=False,compression='gzip')
    role=pd.DataFrame(roles);table(4,role[role.cohort.eq('development')],'new annotations -> core.components');table(5,role[role.cohort.eq('external')],'new annotations -> core.components');table(6,role,'new annotations -> core.components')
    cfg=json.loads((PROJECT/'configs/formal.json').read_text());table(7,pd.DataFrame([{'parameter':k,'value':json.dumps(v)} for k,v in cfg.items()]),'configs/formal.json','Primary CI seed is separately source-verified in configs/statistics.json.')
    all_events=[];all_macro=[];summaries=[];section_rows=[];pair_rows=[]
    for offset,cohort in enumerate(['development','external']):
        for mode in ['','_comparator']:
            p=run/'evaluation'/(cohort+mode+'_events.csv.gz')
            if not p.exists():continue
            ev=pd.read_csv(p);seed=20260812+offset*100003;summary,pair,section=summarize(ev,seed=seed)
            summary.update(cohort=cohort,mode='RadarST' if not mode else 'molecular-score',bootstrap_seed=seed);summaries.append(summary)
            pair['cohort']=cohort;pair['mode']=summary['mode'];section=section.reset_index();section['cohort']=cohort;section['mode']=summary['mode']
            pair_rows.append(pair);section_rows.append(section)
            pair.to_csv(out/(cohort+mode+'_workflow_sample.csv'),index=False);section.to_csv(out/(cohort+mode+'_sample_summary.csv'),index=False)
            pair.groupby('method')[['before','after','gain']].mean().to_csv(out/(cohort+mode+'_workflow_summary.csv'))
            if not mode:
                all_events.append(ev);macro=run/'evaluation'/(cohort+'_macro.csv')
                if macro.exists():all_macro.append(pd.read_csv(macro))
    primary=pd.DataFrame(summaries);primary.to_csv(out/'PRIMARY_RESULTS.csv',index=False)
    if all_events:table(8,pd.concat(all_events,ignore_index=True),'evaluation/{cohort}_events.csv.gz')
    if all_macro:table(9,pd.concat(all_macro,ignore_index=True),'evaluation/{cohort}_macro.csv','SC definition/reference embedding unresolved; not substituted with invented embedding.',status='BLOCKED')
    audits=[];models=[]
    for p in sorted((run/'models').glob('*/TRAINING_AUDIT.json')):
        obj=json.loads(p.read_text());audits.append({'fold':p.parent.name,'held_out':obj['held_out'],'training_datasets':';'.join(obj['training_datasets']),'external_training_rows':obj['external_training_rows'],'heldout_training_rows':obj['heldout_training_rows'],'sampled_rows':json.dumps(obj['sampled_rows']),'hyperparameter_search':obj['hyperparameter_search'],'source':str(p.relative_to(run))})
    model_paths=list((run/'models').glob('*/*.joblib'))+list((run/'experiments/dense/models').glob('*/*.joblib'))
    for p in sorted(model_paths):models.append({'path':str(p.relative_to(run)),'sha256':sha(p),'bytes':p.stat().st_size,'scope':'dense development-only sensitivity' if p.is_relative_to(run/'experiments') else 'primary LOSO/full-fit'})
    table(10,pd.DataFrame(audits),'models/*/TRAINING_AUDIT.json','Fitting isolation is not nested configuration validation.');table(11,pd.DataFrame(models),'models/*/*.joblib')
    pd.DataFrame(models).to_csv(reports/'MODEL_SHA256.csv',index=False)
    sections=pd.concat(section_rows,ignore_index=True) if section_rows else pd.DataFrame();perpair=pd.concat(pair_rows,ignore_index=True) if pair_rows else pd.DataFrame()
    if not sections.empty:table(12,sections[sections.cohort.eq('external')&sections['mode'].eq('RadarST')],'fresh external event hierarchy')
    fromfile(13,run/'experiments/external_lopo/LOPO_RESULTS.csv');fromfile(14,run/'experiments/pathology/CONTEXT_RESULTS.csv')
    exact=[]
    for cohort in ['development','external']:
        p=run/'experiments/pathology'/(cohort+'_context_events.csv.gz')
        if p.exists():
            ev=pd.read_csv(p);metrics=['before','after','gain','coverage_before','coverage_after','purity_before','purity_after','dilution_before','dilution_after']
            frame=ev.groupby(['context','truth_size','dataset','method'])[metrics].mean().groupby(['context','truth_size','dataset']).mean().groupby(['context','truth_size']).mean().reset_index();frame['cohort']=cohort;exact.append(frame)
    if exact:table(15,pd.concat(exact,ignore_index=True),'fresh pathology events -> component/workflow/section equal exact-size summaries')
    if all_events:
        events=pd.concat(all_events,ignore_index=True);den=events.groupby('cohort').agg(sections=('dataset','nunique'),workflow_sample_pairs=('method','size'),unique_truth_components=('truth_component_id','nunique'),events=('gain','size')).reset_index();den['workflow_sample_pairs']=den.cohort.map(perpair[perpair['mode'].eq('RadarST')].groupby('cohort').size());table(16,den,'S8 and fresh pair registry')
    if not sections.empty:
        sections.assign(unit='section').to_csv(reports/'SECTION_RESULTS.csv',index=False);table(17,perpair,'fresh pair/section/workflow hierarchy');sections.to_csv(out/'S17_sections.csv',index=False)
    table(18,primary,'fresh RadarST and comparator events -> equal-section summaries')
    fromfile(19,run/'experiments/randomization/RANDOMIZATION_RESULTS.csv');fromfile(20,run/'experiments/sensitivity/GRID_RESULTS.csv')
    fromfile(21,run/'experiments/proposal/ORACLE_COMPONENTS.csv.gz');fromfile(22,run/'experiments/proposal/SATURATION_CURVE.csv');fromfile(23,run/'experiments/proposal/CANDIDATE_COUNTS.csv')
    fromfile(24,run/'biology/GENE_CONCORDANCE.csv');fromfile(25,run/'biology/intermediate/cfar_pathway_classification_master.csv');fromfile(26,run/'biology/cfar_random_calibration.csv')
    for n in range(1,27):
        if not any(r['table']==f'S{n}' for r in registry):registry.append({'table':f'S{n}','title':TABLE_NAMES[n],'status':'BLOCKED','path':'','rows':0,'source':'','notes':'Prerequisite output unavailable','sha256':''})
    pd.DataFrame(registry).to_csv(reports/'SUPPLEMENT_TABLE_STATUS.csv',index=False)
    for cohort,name in [('development','DEVELOPMENT_RESULTS.csv'),('external','EXTERNAL_RESULTS.csv')]:primary[primary.cohort.eq(cohort)&primary['mode'].eq('RadarST')].to_csv(reports/name,index=False)
    primary.to_csv(reports/'COMPARATOR_RESULTS.csv',index=False)
    paired_comparator(run,all_events,primary)
    write_md(reports/'STATISTICAL_AUDIT.md','# Statistical audit\n\nPrimary hierarchy: mean components within each workflow-section, mean workflows within each section, mean independent sections. 10,000 section bootstrap draws; seed 20260812 and external offset 100003. Exact-size absent strata remain missing. Event fractions are descriptive, not independent-sample tests. Risk comparison uses paired section bootstrap. Pathology CI uses 2,000 section draws. No confidence-interval positivity is forced.\n\nThe earlier orchestration default seed 20260720 was superseded for final primary tables by the documented 79A source seed, not by numerical optimization. Model fitting configuration is unchanged.\n')
    write_md(reports/'TRAINING_REPRODUCTION_REPORT.md','# Training reproduction\n\n'+pd.DataFrame(audits).to_string(index=False)+'\n\nSee model hashes and TRAINING_INPUTS.csv for actual training provenance. Annotation-derived targets are supervised Development inputs; held-out/external targets are excluded from each fitting table. Configuration was inherited from development history; this is not nested model selection.\n')
    import inspect
    lineage=pd.read_csv(PROJECT/'manifests/SOURCE_FUNCTION_LINEAGE.csv');wrappers=[]
    for name in ['read_expression','feature_pcs','preprocess_coordinates','generate','features','teacher_targets','utility_targets','infer','evaluate','components']:
        fn=getattr(core,name);wrappers.append({'function':name,'file':'src/radarst/core.py','line':inspect.getsourcelines(fn)[1]})
    pd.DataFrame(wrappers).to_csv(reports/'METHODS_WRAPPER_LINES.csv',index=False)
    write_md(reports/'METHODS_CODE_MAPPING.md','# Methods/code mapping\n\n'+pd.DataFrame(wrappers).to_string(index=False)+'\n\n'+lineage.to_string(index=False)+'\n\nStable full candidate SHA256 and strict one-to-one joins replace randomized truncated IDs. This is procedure reproduction after an identity correctness repair, not historical bitwise replay.\n')


def paired_comparator(run,events,primary):
    rows=[]
    for offset,cohort in enumerate(['development','external']):
        p=run/'evaluation'/(cohort+'_events.csv.gz');q=run/'evaluation'/(cohort+'_comparator_events.csv.gz')
        if not p.exists() or not q.exists():continue
        a=pd.read_csv(p);b=pd.read_csv(q);key=['dataset','method','truth_component_id']
        joined=a[a.truth_size.between(1,10)].merge(b[b.truth_size.between(1,10)],on=key,suffixes=('_radarst','_molecular'),validate='one_to_one')
        joined['loss_difference']=joined.gain_radarst.lt(-1e-12).astype(float)-joined.gain_molecular.lt(-1e-12).astype(float);joined['gain_difference']=joined.gain_radarst-joined.gain_molecular
        per=joined.groupby(['dataset','method'])[['loss_difference','gain_difference']].mean().groupby('dataset').mean();rng=np.random.default_rng(20260812+offset*100003)
        indices=rng.integers(0,len(per),(10000,len(per)))
        for metric in per:
            values=per[metric].to_numpy();draw=values[indices].mean(1);lo,hi=np.quantile(draw,[.025,.975]);rows.append({'cohort':cohort,'metric':'RadarST-minus-molecular-'+metric,'estimate':values.mean(),'ci_low':lo,'ci_high':hi,'independent_sections':len(values),'bootstrap':10000})
    pd.DataFrame(rows).to_csv(run/'reports/PAIRED_COMPARATOR.csv',index=False)


def figures(run):
    from .plots import build
    build(run)


def verify(run):
    reports=run/'reports';reports.mkdir(exist_ok=True);samples,pairs=manifests();checks=[]
    def check(name,passed,detail):checks.append({'check':name,'status':'PASS' if passed else 'BLOCKED','detail':detail})
    predictions=list((run/'pairs').glob('*/*/inference/final_labels.csv'));check('322 inference outputs',len(predictions)==322,len(predictions))
    output_errors=[]
    for r in pairs[pairs.cohort.ne('control')].itertuples():
        folder=path(run,r.dataset,r.method)/'inference'
        try:
            manifest=json.loads((folder/'SEAL.json').read_text())
            if not trusted(folder,manifest['inputs']):raise RuntimeError('Output seal mismatch')
            if manifest['inputs']['features']!=sha(path(run,r.dataset,r.method)/'features/features.parquet'):raise RuntimeError('Feature input changed')
            md=run/'models'/(r.dataset if r.cohort=='development' else 'full_fit')
            if manifest['inputs']['selector_model']!=sha(md/'selector_heads.joblib') or manifest['inputs']['ranker_model']!=sha(md/'expected_gain.joblib'):raise RuntimeError('Model input changed')
            frame=pd.read_csv(folder/'final_labels.csv',dtype=str)
            coord=pd.read_csv(PROJECT/'data/coordinates'/f'{r.dataset}.csv',dtype={'spot_id':str})
            if not frame.spot_id.is_unique or frame.spot_id.tolist()!=coord.spot_id.tolist() or frame.pred_label.isna().any():raise RuntimeError('Final label identity/completeness failure')
        except Exception as exc:output_errors.append({'dataset':r.dataset,'method':r.method,'error':str(exc)})
    check('322 current five-layer inference seals',not output_errors,output_errors)
    models=list((run/'models').glob('*/selector_heads.joblib'));check('12 LOSO plus full fit',len(models)==13,len(models))
    for p in (run/'models').glob('*/TRAINING_AUDIT.json'):
        a=json.loads(p.read_text());check('training isolation '+p.parent.name,a['heldout_training_rows']==0 and a['external_training_rows']==0,a['training_datasets'])
        forbidden={'candidate_truth_best_IoU','candidate_truth_component_id','base_best_IoU_for_candidate_target','signed_utility_target','utility_target','teacher_E7_selected'}
        columns=a['feature_columns'];check('104 target-blind ranker features '+p.parent.name,len(columns)==104 and not forbidden.intersection(columns),'Explicit ordered source feature list; supervised targets are not predictor columns')
        try:verified_models(run,p.parent,'FULL' if p.parent.name=='full_fit' else p.parent.name);valid=True;detail='All target/model hashes current'
        except Exception as exc:valid=False;detail=str(exc)
        check('current model training provenance '+p.parent.name,valid,detail)
    for p in (run/'samples').glob('*/candidates/pool.parquet'):
        frame=pd.read_parquet(p);check('candidate uniqueness '+p.parent.parent.name,frame.candidate_id.is_unique and frame.member_spots.is_unique,len(frame))
    table_status=pd.read_csv(reports/'SUPPLEMENT_TABLE_STATUS.csv');check('S1-S26 complete',len(table_status)==26 and table_status.status.isin(['FRESH','VERIFIED']).all(),table_status[~table_status.status.isin(['FRESH','VERIFIED'])].table.tolist())
    figure_manifest=reports/'FIGURE_SOURCE_MANIFEST.csv'
    if figure_manifest.exists():
        fm=pd.read_csv(figure_manifest);check('main/supplement figures complete',fm.figure.nunique()==10 and fm.status.eq('FRESH').all(),fm.figure.nunique())
    else:check('main/supplement figures complete',False,'No source manifest')
    install=reports/'CLEAN_ENV_QA.json'
    if not install.exists() and not output_errors:
        python=PROJECT/'.venv/Scripts/python.exe'
        if python.exists():
            command=[str(python),str(PROJECT/'scripts/test_independent_inference.py'),'--run',run.name,'--output-name','independent_clean_environment']
            result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace')
            write_md(run/'logs/CLEAN_INFERENCE.log',result.stdout+'\n'+result.stderr)
            dump(reports/'CLEAN_INFERENCE_COMMAND.json',{'command':command,'exit_code':result.returncode})
    qa=json.loads(install.read_text()) if install.exists() else {};check('clean install and minimal inference',qa.get('passed') is True and qa.get('run')==run.name and Path(qa.get('prefix','')).resolve()==(PROJECT/'.venv').resolve(),'Separate current-run clean environment verification required')
    policy=json.loads((PROJECT/'configs/annotation_policy.json').read_text())['development_truth_fields'];field_rows=[]
    for s in samples[samples.cohort.eq('development')].itertuples():
        raw=pd.read_csv(PROJECT/s.metadata,sep='\t' if s.metadata.endswith('.tsv') else ',',dtype=str,keep_default_na=False)
        barcode=next(c for c in ['spot_id','ID','barcode','spot_id_normalized','Unnamed: 0'] if c in raw)
        ann=pd.read_csv(PROJECT/'data/annotations'/f'{s.dataset}.csv',dtype=str,keep_default_na=False)
        expected=raw.set_index(barcode)[policy[s.dataset]].reindex(ann.spot_id)
        if s.dataset=='A01':expected=expected.replace({'Uncertain':'UNASSIGNED','Uncertain_overlap':'UNASSIGNED'})
        field_rows.append({'dataset':s.dataset,'field':policy[s.dataset],'passed':expected.tolist()==ann.truth_label.tolist(),'annotation_sha256':sha(PROJECT/'data/annotations'/f'{s.dataset}.csv')})
    fa=pd.DataFrame(field_rows);fa.to_csv(reports/'CURRENT_ANNOTATION_FIELD_AUDIT.csv',index=False);correct_fields=len(fa)==12 and fa.passed.all()
    check('declared annotation fields',correct_fields,'Source-declared fine annotations, barcode-wise verified and current annotation hashes matched')
    version=PROJECT/'manifests/DLPFC_151673_VERSION_AUDIT.csv';va=pd.read_csv(version) if version.exists() else pd.DataFrame()
    check('DLPFC 151673 imported label provenance',len(va)==14 and va.exact_label_match_to_documented_upstream_source.all(),'Input provenance independently aligned to documented upstream export, not inferred from ARI')
    check('DLPFC historical metric version correspondence',False,'Four previously reported metric/input versions remain unresolved; current 168 metrics are freshly calculated from declared maps')
    check('SC reference embedding',False,'Formal source definition and input embedding not independently recovered; no substitute silently used')
    check('redistribution licenses',False,'Code and third-party data/model redistribution rights require author confirmation')
    exportqa=PROJECT/'reports/PUBLIC_EXPORT_QA.json';pub=json.loads(exportqa.read_text()) if exportqa.exists() else {}
    check('GitHub code-only export scan',bool(pub) and not pub.get('findings') and pub.get('raw_data_included') is False and pub.get('weights_included') is False,'Local code-only export scan; does not establish code/data redistribution rights')
    regression=reports/'INDEPENDENT_INFERENCE_REGRESSION.csv';rg=pd.read_csv(regression) if regression.exists() else pd.DataFrame()
    check('single-upstream independent deployment',len(rg)==10 and rg.passed.all(),'Actual two-pair, five-layer raw-to-final regression; not inferred from benchmark inference')
    command=[sys.executable,str(PROJECT/'scripts/audit_results_independently.py'),'--run',run.name]
    independent=subprocess.run(command,cwd=PROJECT,env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace')
    write_md(run/'logs/INDEPENDENT_STATISTICS.log',independent.stdout+'\n'+independent.stderr)
    check('independent statistical recalculation',independent.returncode==0,'Separate implementation of both primary cohorts/comparator, section bootstrap CI, pair signs and event fractions; tolerance 1e-12')
    claim_command=[sys.executable,str(PROJECT/'scripts/report_scientific_claims.py'),'--run',run.name]
    claims=subprocess.run(claim_command,cwd=PROJECT,env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace');write_md(run/'logs/SCIENTIFIC_CLAIMS.log',claims.stdout+'\n'+claims.stderr)
    check('scientific claim registry generated',claims.returncode==0,'A generated registry is not a claim that every scientific conclusion passed')
    cost_command=[sys.executable,str(PROJECT/'scripts/audit_comparator_cost.py'),'--run',run.name]
    cost=subprocess.run(cost_command,cwd=PROJECT,env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace')
    write_md(run/'logs/COMPARATOR_COST_AUDIT.log',cost.stdout+'\n'+cost.stderr)
    check('322 comparator slot/cost audit',cost.returncode==0,'Same pre-retention slots and retention rule; actual atomic packing differences are reported, not claimed identical')
    primary=pd.read_csv(run/'tables/PRIMARY_RESULTS.csv')
    development_gain=float(primary[primary.cohort.eq('development')&primary['mode'].eq('RadarST')].gain.iloc[0])
    grid=pd.read_csv(run/'experiments/sensitivity/GRID_RESULTS.csv')
    anchor=grid[np.isclose(grid.expression_weight,.80)&np.isclose(grid.retention,.97)]
    check('sensitivity source anchor equals primary',len(anchor)==1 and abs(float(anchor.gain.iloc[0])-development_gain)<=1e-12,'0.80/0.20/0.025/0.97, common source Development RNG tag; no grid-based model selection')
    random_status=pd.read_csv(run/'experiments/randomization/PAIR_STATUS.csv')
    check('322 complete current candidate randomizations',len(random_status)==322 and random_status.status.isin(['FRESH','TRUSTED_CURRENT_RUN']).all(),'1,000 realizations per pair, exact count/burden/channel/type/size and nonoverlap assertions in generation')
    dense_summary=pd.read_csv(run/'experiments/dense/SUMMARY.csv')
    dense_anchor=dense_summary[dense_summary.variant.eq('S0_FORMAL')]
    check('dense S0 summary equals primary',len(dense_anchor)==1 and abs(float(dense_anchor.gain.iloc[0])-development_gain)<=1e-12,'Exploratory dense variants must retain the actual primary control')
    dense_regression=[]
    from sklearn.metrics import adjusted_rand_score
    for r in pairs[pairs.cohort.eq('development')].itertuples():
        primary_folder=path(run,r.dataset,r.method)/'inference';dense_folder=run/'experiments/dense/predictions/S0_FORMAL'/r.dataset/r.method
        a=pd.read_csv(primary_folder/'selected_regions.csv');b=pd.read_csv(dense_folder/'regions.csv')
        x=pd.read_csv(primary_folder/'final_labels.csv',dtype=str).set_index('spot_id').pred_label;y=pd.read_csv(dense_folder/'labels.csv',dtype=str).set_index('spot_id').pred_label
        same_members=set(a.member_spots)==set(b.member_spots);same_partition=set(x.index)==set(y.index) and adjusted_rand_score(x,y.reindex(x.index))>=1-1e-12
        dense_regression.append({'dataset':r.dataset,'method':r.method,'same_retained_member_sets':same_members,'same_final_partition':same_partition})
    dense_regression=pd.DataFrame(dense_regression);dense_regression.to_csv(reports/'DENSE_S0_PRIMARY_REGRESSION.csv',index=False)
    check('168 dense S0 retained-region/partition controls',len(dense_regression)==168 and dense_regression.same_retained_member_sets.all() and dense_regression.same_final_partition.all(),'Label-string prefixes may differ; biological partition and complete selected memberships must match')
    biology_command=[sys.executable,str(PROJECT/'scripts/audit_biology_classification.py'),'--run',run.name]
    biology_audit=subprocess.run(biology_command,cwd=PROJECT,env=dict(os.environ),capture_output=True,text=True,encoding='utf-8',errors='replace')
    write_md(run/'logs/BIOLOGY_CLASSIFICATION_AUDIT.log',biology_audit.stdout+'\n'+biology_audit.stderr)
    check('biological classification audit execution',biology_audit.returncode==0,'Audit reruns from current outputs; it does not alter inherited pathway classifications')
    biological=reports/'CONTEXT_CLASSIFICATION_AUDIT.csv'
    if biological.exists():
        bi=pd.read_csv(biological);consistent=np.maximum(bi.finite_fraction_positive,1-bi.finite_fraction_positive).ge(.75)&bi.finite_attribution_regions.ge(8)&bi.median_attribution.abs().ge(.05)
        check('context classification missing-value sensitivity',bool(consistent.all()),bi.loc[~consistent,['library','term','context','missing_fraction','finite_fraction_positive']].to_dict('records'))
    else:check('context classification missing-value sensitivity',False,'Actual source criterion and finite-data availability audit not completed')
    pd.DataFrame(checks).to_csv(reports/'RELEASE_CHECKS.csv',index=False)
    stages=[]
    for p in (run/'logs').glob('stage_*.json'):stages.append(json.loads(p.read_text()))
    core_done=len(predictions)==322 and len(models)==13 and not output_errors and all(r['status']=='PASS' for r in checks if r['check'].startswith(('training isolation','current model training provenance','candidate uniqueness','104 target-blind')))
    category='A' if all(r['status']=='PASS' for r in checks) else 'B' if core_done else 'C'
    issues=[r for r in checks if r['status']!='PASS']
    science=json.loads((reports/'SCIENTIFIC_CHANGE_STATUS.json').read_text()) if (reports/'SCIENTIFIC_CHANGE_STATUS.json').exists() else {'scientific_change_marker':'NOT VERIFIED'}
    dump(run/'COMPUTATION_SEAL.json',{'sealed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'current generated outputs; incomplete steps explicitly recorded','historical_inputs_read_by_pipeline':False,'execution_category':category,'scientific_change_marker':science['scientific_change_marker']})
    hashes=[]
    for p in run.rglob('*'):
        if p.is_file() and p.name!='OUTPUT_SHA256.csv':hashes.append({'path':str(p.relative_to(run)),'sha256':sha(p),'bytes':p.stat().st_size})
    pd.DataFrame(hashes).to_csv(reports/'OUTPUT_SHA256.csv',index=False)
    write_md(reports/'REMAINING_ISSUES.md','# Remaining issues\n\n'+pd.DataFrame(issues).to_string(index=False)+'\n')
    write_md(reports/'GITHUB_RELEASE_QA.md','# GitHub release QA\n\nNOT READY. No upload or push performed. Raw data, vendor binaries, intermediate caches and provenance containing workstation paths are excluded by .gitignore; no code license has been invented. Clean installation, public-input licensing, public export privacy scan and source completeness must pass before release.\n\n'+pd.DataFrame(checks).to_string(index=False)+'\n')
    write_md(reports/'FULL_CLEAN_REPRODUCTION_REPORT.md',f'# Clean reproduction report\n\nExecution category: **{category}**. Scientific change marker: **{science["scientific_change_marker"]}**. These are separate classifications. This does not imply all manuscript conclusions or all supplements passed.\n\n'+pd.DataFrame(stages)[['stage','status','seconds']].to_string(index=False)+'\n\n'+pd.DataFrame(checks).to_string(index=False)+'\n\nHistorical outputs have not been used to fill failed stages. Old-versus-new comparison must be performed in a separate post-seal process.\n')
    write_md(reports/'MANUSCRIPT_REVISION_CHECKLIST.md','# Manuscript revision checklist\n\nOriginal manuscript remains unchanged. Reconcile all fresh primary, comparator, pathology and biological findings only after independent old/new comparison. Do not substitute procedure reproduction for historical exact-output reproduction. Do not claim complete SC, upstream-device provenance, or released code while unresolved.\n')
    if issues:raise RuntimeError('Release verification has blocking checks; see reports/RELEASE_CHECKS.csv')
