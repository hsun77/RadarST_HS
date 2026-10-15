"""Original supervised targets/fitting with explicit user-owned training inputs."""
from pathlib import Path
from types import SimpleNamespace
import json
import os


TYPES=["core_10","core_3","core_4","core_5","core_8","edge_pair","singleton"]


def resolve(base,value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base/p).resolve()


def prepare(manifest_path,output,teacher_source='provided',held_out=None,audit_access=False):
    import numpy as np
    import pandas as pd
    import yaml
    from threadpoolctl import threadpool_limits
    from .config import new_output,DEFAULTS
    from .inference import input_row,setup
    from .engine import training_functions
    if os.environ.get("PYTHONHASHSEED")!="0":
        raise ValueError("Set PYTHONHASHSEED=0 before starting Python")
    path=Path(manifest_path).resolve()
    data=yaml.safe_load(path.read_text())
    all_samples=data.get('samples',[])
    if not all_samples or any(s.get('cohort')!='development' for s in all_samples):
        raise ValueError('Only explicitly development samples may enter training')
    if held_out is not None and held_out not in {s['dataset'] for s in all_samples}:
        raise ValueError('Held-out dataset must be declared')
    samples=[s.copy() for s in all_samples if s['dataset']!=held_out]
    if not samples:
        raise ValueError('No included development samples')
    # Check mandatory supervision BEFORE creating outputs or computing features.
    for sample in samples:
        if sample.get("cohort")!="development":
            raise ValueError("Only explicitly development samples may enter training")
        if teacher_source=='provided' and not sample.get("teacher_acceptance"):
            raise ValueError("BLOCKED: original-format teacher_acceptance supervision is required; reference overlap is not a substitute")
        if teacher_source=='generate' and len(sample.get('upstream',[]))!=14:
            raise ValueError('The original teacher requires exactly 14 upstream workflows per included sample')
        fields=['expression','coordinates','reference']+(['teacher_acceptance'] if teacher_source=='provided' else [])
        for field in fields:
            if not resolve(path.parent,sample[field]).is_file():
                raise ValueError(f"Missing training input: {field}")
    if len({s['dataset'] for s in samples})!=len(samples):
        raise ValueError("Training sample IDs must be unique")
    from .teacher import selected_samples
    selected_samples(path,held_out)
    out=new_output(output)
    scratch=out/"runtime_tmp";scratch.mkdir()
    os.environ.update(NUMBA_CACHE_DIR=str(out/"numba_cache"),MPLCONFIGDIR=str(out/"mplconfig"),TEMP=str(scratch),TMP=str(scratch))
    log=None
    if audit_access:
        from .audit import install
        allowed=[path,Path(__file__).with_name('teacher_protocol.json')]
        for sample in samples:
            allowed += [resolve(path.parent,sample[k]) for k in ['expression','coordinates','reference','teacher_acceptance','feature_expression'] if sample.get(k) and (k!='teacher_acceptance' or teacher_source=='provided')]
            allowed += [resolve(path.parent,m['labels']) for m in sample.get('upstream',[])]
        log=install(out,allowed)
    tf=training_functions()
    records=[]
    for sample in samples:
        d=sample['dataset']
        folder=out/d;folder.mkdir()
        if teacher_source=='generate':
            from .teacher import generate_one
            sample['teacher_acceptance']=str(generate_one(sample,path.parent,folder/'teacher'))
        ref=pd.read_csv(resolve(path.parent,sample['reference']),dtype=str,keep_default_na=False)
        if not {"spot_id","truth_label","truth_component_id"}.issubset(ref) or ref.spot_id.duplicated().any():
            raise ValueError("Training reference requires unique spot_id,truth_label,truth_component_id; component IDs preserve target tie behaviour")
        if ref.truth_label.isin(["Uncertain","Uncertain_overlap"]).any():
            raise ValueError("Normalize excluded uncertainty labels to UNASSIGNED before training")
        components=[]
        valid=ref.loc[~ref.truth_label.str.lower().isin({"unassigned","unknown","nan","","none"}) & ref.truth_component_id.ne("")]
        for cid,group in valid.groupby('truth_component_id',sort=False):
            components.append(dict(truth_component_id=cid,truth_spots=group.spot_id.tolist(),component_size_spots=len(group)))
        if not components:
            raise ValueError("No eligible reference components")
        comp_sets,_,spot_to_components=tf.a.component_maps(pd.DataFrame(components))
        teacher_path=resolve(path.parent,sample['teacher_acceptance'])
        tf.keys.load_teacher_keys.__globals__['TEACHER']=teacher_path
        teacher_keys=tf.keys.load_teacher_keys()
        if any(key[0]!=d for key in teacher_keys):
            raise ValueError("Each teacher acceptance file must belong only to its declared sample")
        candidates=ctx=indices=None
        with threadpool_limits(limits=20,user_api='blas'),threadpool_limits(limits=1,user_api='openmp'):
            for upstream in sample.get('upstream',[]):
                m=upstream['workflow']
                dest=folder/m;dest.mkdir()
                args=SimpleNamespace(expression=str(resolve(path.parent,sample['expression'])),coordinates=str(resolve(path.parent,sample['coordinates'])),upstream_labels=str(resolve(path.parent,upstream['labels'])),dataset=d,workflow=m,barcode_column=upstream.get('barcode_column','spot_id'),label_column=upstream.get('label_column','pred_label'),feature_expression=str(resolve(path.parent,sample['feature_expression'])) if sample.get('feature_expression') else None,feature_kind=sample.get('feature_kind','h5ad'),allow_positional_upstream=False)
                cfg=DEFAULTS.copy();cfg['profile']='internal'
                row=input_row(args,dest,cfg)
                F,f=setup(row)
                if candidates is None:
                    candidates,ctx,indices,_=F.generate_sample(row,f)
                frame,base=F.build_pair_features(row,candidates,ctx,indices,f,dest)
                if not set(ctx['spot_ids']).issubset(set(ref.spot_id)):
                    raise ValueError("Missing training reference spots")
                overlap,cids=tf.a.candidate_truth_overlap(frame,comp_sets,spot_to_components)
                regions,_,_=f.b67.label_connected_components(base,ctx['spot_ids'],ctx['adjacency'])
                from .engine import _load
                metrics=_load('metrics')
                baseline={cid:metrics.true_best_match_from_regions(regions,members)['iou'] for cid,members in comp_sets.items()}
                frame['candidate_truth_best_IoU']=overlap
                frame['candidate_truth_component_id']=cids
                frame['base_best_IoU_for_candidate_target']=[baseline.get(str(cid),0.) for cid in cids]
                frame['utility_target']=np.maximum(0.,overlap-frame.base_best_IoU_for_candidate_target.to_numpy(float))
                frame['signed_utility_target']=overlap-frame.base_best_IoU_for_candidate_target.to_numpy(float)
                frame['teacher_E7_selected']=[int((d,m,str(key)) in teacher_keys) for key in frame.candidate_key]
                target=dest/'training_table.parquet'
                frame.to_parquet(target,index=False)
                records.append(dict(dataset=d,workflow=m,cohort='development',table=str(target.relative_to(out)),sha256=F.sha(target),teacher_input_sha256=F.sha(teacher_path),reference_sha256=F.sha(resolve(path.parent,sample['reference']))))
    if not records:
        raise ValueError("No upstream training maps were declared")
    (out/'training_tables.yaml').write_text(yaml.safe_dump({'tables':records,'excluded_dataset_before_target_construction':held_out,'teacher_source':teacher_source},sort_keys=False))
    if log: log.flush()
    print(out/'training_tables.yaml')


def load_fold(records,base,included):
    import pandas as pd
    parts=[]
    from .frontend import sha
    for record in sorted(records,key=lambda r: Path(r['dataset'])/(r['workflow']+'.parquet')):
        if record['dataset'] not in included:
            continue
        path=resolve(base,record['table'])
        if sha(path)!=record['sha256']:
            raise ValueError("Training table checksum mismatch")
        frame=pd.read_parquet(path)
        if not frame.dataset_id.eq(record['dataset']).all() or not frame.method.eq(record['workflow']).all():
            raise ValueError("Training table sample/workflow mismatch")
        parts.append(frame)
    if not parts:
        raise ValueError("No included development training tables")
    return parts


def train(config_path):
    import numpy as np
    import pandas as pd
    import yaml
    import joblib
    from threadpoolctl import threadpool_limits
    from .config import new_output
    from .engine import training_functions
    from .frontend import sha
    path=Path(config_path).resolve()
    cfg=yaml.safe_load(path.read_text())
    manifest=resolve(path.parent,cfg['manifest'])
    prepared_manifest=yaml.safe_load(manifest.read_text())
    records=prepared_manifest.get('tables',[])
    if not records or any(r.get('cohort')!='development' for r in records):
        raise ValueError("Only explicit prepared development tables can be fitted; external truth is forbidden")
    datasets=sorted({r['dataset'] for r in records})
    mode=cfg.get('mode','loso_and_full')
    excluded=cfg.get('held_out_dataset')
    if excluded is not None and (mode!='full' or excluded in datasets or prepared_manifest.get('excluded_dataset_before_target_construction')!=excluded):
        raise ValueError('Fold-specific raw-input training requires exclusion before target construction')
    if mode not in {'loso','full','loso_and_full'} or mode!='full' and len(datasets)<2:
        raise ValueError("LOSO requires at least two independent sample identifiers")
    out=new_output(resolve(path.parent,cfg['output']))
    os.environ.update(NUMBA_CACHE_DIR=str(out/'numba_cache'),MPLCONFIGDIR=str(out/'mplconfig'))
    f=training_functions()
    folds=datasets if mode in {'loso','loso_and_full'} else []
    if mode in {'full','loso_and_full'}:folds=folds+[None]
    for heldout in folds:
        included=set(datasets)-({heldout} if heldout else set())
        parts=load_fold(records,manifest.parent,included)
        required={'teacher_E7_selected','utility_target','signed_utility_target','candidate_truth_best_IoU','base_best_IoU_for_candidate_target'}
        for part in parts:
            if not required.issubset(part) or not part.teacher_E7_selected.isin([0,1]).all():
                raise ValueError("Missing/invalid original supervised targets")
            signed=part.candidate_truth_best_IoU.to_numpy(float)-part.base_best_IoU_for_candidate_target.to_numpy(float)
            if not np.array_equal(signed,part.signed_utility_target.to_numpy(float)) or not np.array_equal(np.maximum(0,signed),part.utility_target.to_numpy(float)):
                raise ValueError("Supervised target values do not match the original definitions")
        teacher=pd.concat([f.d.deterministic_training_sample(p) for p in parts],ignore_index=True)
        signed=pd.concat([f.a.training_sample(p) for p in parts],ignore_index=True)
        rank_training=pd.concat([f.harm.sample_frame(p) for p in parts],ignore_index=True)
        if set(teacher.teacher_E7_selected)!={0,1} or not (signed.signed_utility_target>0).any() or not (rank_training.signed_utility_target<0).any():
            raise ValueError("Both teacher classes and positive/harm examples are required for compatible models")
        assert heldout not in set(teacher.dataset_id) and heldout not in set(signed.dataset_id) and heldout not in set(rank_training.dataset_id)
        x=f.inference.g.feature_frame(rank_training,TYPES)
        if x.shape[1]!=104 or not np.isfinite(x.to_numpy()).all():
            raise ValueError("Training features do not match the original 104-feature schema")
        with threadpool_limits(limits=20,user_api='blas'),threadpool_limits(limits=1,user_api='openmp'):
            teacher_model=f.t.fit_model(teacher,TYPES)
            classifier,regressor=f.signed.fit_signed_heads(signed,TYPES)
            rank=f.rank.fit_models(rank_training,TYPES)
        common=dict(held_out_dataset=excluded if excluded is not None else heldout,training_datasets=sorted(included),supervised_protocol='original teacher/signed/groupwise fitting definitions')
        heads=dict(teacher=teacher_model,signed_utility_classifier=classifier,signed_utility_regressor=regressor,proposal_types=TYPES,harm_weight=.5,extra_utility_budget=.01,**common)
        rank.update(common)
        dest=out/('full' if heldout is None else 'LOSO_holdout_'+heldout);dest.mkdir()
        joblib.dump(heads,dest/'selector_heads.joblib',compress=3)
        joblib.dump(rank,dest/'expected_gain.joblib',compress=3)
        (dest/'training_metadata.json').write_text(json.dumps(dict(**common,teacher_rows=len(teacher),signed_rows=len(signed),rank_rows=len(rank_training),heldout_rows=0,external_rows=0,manifest_sha256=sha(manifest),selector_sha256=sha(dest/'selector_heads.joblib'),ranker_sha256=sha(dest/'expected_gain.joblib')),indent=2))
        print(dest)


def train_inputs(manifest,output,mode='loso_and_full',held_out=None,audit_access=False):
    """Run fold-isolated original preparation and fitting in separate processes."""
    import datetime
    import subprocess
    import sys
    import yaml
    from .teacher import selected_samples
    from .config import new_output
    path,samples=selected_samples(manifest,held_out)
    ids=[s['dataset'] for s in yaml.safe_load(path.read_text())['samples']]
    if held_out is not None and held_out not in ids:
        raise ValueError('Held-out dataset must be declared')
    if mode not in {'loso','full','loso_and_full'} or mode!='full' and len(ids)<2:
        raise ValueError('LOSO requires at least two development samples')
    if any(len(s.get('upstream',[]))!=14 for s in samples if s['dataset']!=held_out):
        raise ValueError('Original teacher training requires 14 maps per included development sample')
    if os.environ.get('PYTHONHASHSEED')!='0':
        raise ValueError('Set PYTHONHASHSEED=0 before starting Python')
    if held_out is not None and mode!='loso':
        raise ValueError('--held-out requires --mode loso')
    out=new_output(output)
    folds=([held_out] if held_out else sorted(ids)) if mode!='full' else []
    if mode in {'full','loso_and_full'}: folds.append(None)
    for held in folds:
        dest=out/('full' if held is None else 'LOSO_holdout_'+held); dest.mkdir()
        commands=[]
        def execute(argv,label):
            start=datetime.datetime.now(datetime.timezone.utc).isoformat()
            with (dest/(label+'.log')).open('w') as stream:
                result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            commands.append(dict(argv=argv,exit_code=result.returncode,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            (dest/'commands.json').write_text(json.dumps(commands,indent=2))
            if result.returncode:
                raise RuntimeError('Training stage failed; inspect '+str(dest/(label+'.log')))
        command=[sys.executable,'-m','radarst','prepare-training','--manifest',str(path),
            '--output',str(dest/'prepared'),'--teacher-source','generate']
        if held is not None: command+=['--held-out',held]
        if audit_access: command+=['--audit-access']
        execute(command,'prepare')
        config=dict(manifest=str(dest/'prepared/training_tables.yaml'),output=str(dest/'models'),mode='full')
        if held is not None: config['held_out_dataset']=held
        conf=dest/'fit.yaml';conf.write_text(yaml.safe_dump(config))
        execute([sys.executable,'-m','radarst','train','--config',str(conf)],'fit')
    print(out)
