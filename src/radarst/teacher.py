"""Raw-input adapter for the original 56C -> 57D -> 70B -> 71B teacher.

Only orchestration and file adapters live here. Computational functions are
mechanically extracted originals, including the short-ID left join.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace


def selected_samples(manifest, held_out=None):
    import yaml
    from .training import resolve
    path = Path(manifest).resolve()
    data = yaml.safe_load(path.read_text())
    samples = data.get('samples', [])
    if not samples or any(s.get('cohort') != 'development' for s in samples):
        raise ValueError('Teacher/training inputs must be explicitly development samples; external inputs are forbidden')
    ids = [s['dataset'] for s in samples]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', d) or '..' in d for d in ids):
        raise ValueError('Training dataset IDs must be unique safe path components')
    if held_out is not None and held_out not in ids:
        raise ValueError('Held-out dataset must be declared in the manifest')
    included = [s for s in samples if s['dataset'] != held_out]
    if not included:
        raise ValueError('No included development samples')
    # Exclusion precedes existence checks, map reads, teacher and target generation.
    for sample in included:
        maps = sample.get('upstream', [])
        names = [m['workflow'] for m in maps]
        if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', m) or '..' in m for m in names):
            raise ValueError('Workflow IDs must be unique safe path components')
        for field in ['expression', 'coordinates']:
            if not resolve(path.parent, sample[field]).is_file():
                raise ValueError('Missing teacher input: '+field)
    return path, included


def generate_one(sample, base, output):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from .training import resolve
    from .config import DEFAULTS
    from .inference import input_row, setup
    from .engine import _load, training_functions
    from .frontend import sha

    maps = sample.get('upstream', [])
    if len(maps) != 14:
        raise ValueError('The original teacher requires exactly 14 distinct upstream workflows per included sample (fixed support denominator 14)')
    output = Path(output)
    output.mkdir()
    inputs = {}
    for field in ['expression', 'coordinates', 'feature_expression']:
        if sample.get(field):
            p = resolve(base, sample[field])
            inputs[field] = dict(path=str(p), sha256=sha(p))
    rows = []
    F = f = candidates = ctx = None
    cfg = DEFAULTS.copy(); cfg['profile'] = 'internal'
    with threadpool_limits(limits=20, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
        for i, upstream in enumerate(maps):
            folder = output/('map_'+str(i)); folder.mkdir()
            labels = resolve(base, upstream['labels'])
            args = SimpleNamespace(expression=str(resolve(base, sample['expression'])),
                coordinates=str(resolve(base, sample['coordinates'])), upstream_labels=str(labels),
                dataset=sample['dataset'], workflow=upstream['workflow'],
                barcode_column=upstream.get('barcode_column','spot_id'),
                label_column=upstream.get('label_column','pred_label'),
                feature_expression=str(resolve(base,sample['feature_expression'])) if sample.get('feature_expression') else None,
                feature_kind=sample.get('feature_kind','h5ad'),allow_positional_upstream=False)
            row = input_row(args,folder,cfg)
            if candidates is None:
                F,f = setup(row)
                candidates,ctx,_,scores = F.generate_sample(row,f)
                candidates.to_csv(output/'fresh_proposals.csv',index=False)
                pd.DataFrame(dict(spot_id=ctx['spot_ids'],CFAR=scores)).to_csv(output/'fresh_CFAR.csv',index=False)
            rows.append(dict(sample_id=sample['dataset'],method=upstream['workflow'],
                pred_path=str(labels),barcode_col=args.barcode_column,pred_label_col=args.label_column))
            inputs['map_'+str(i)] = dict(path=str(labels),sha256=sha(labels))
        canonical = pd.DataFrame(dict(sample_id=sample['dataset'],
            spot_id_normalized=ctx['spot_ids'],invalid_coordinate_spot=False,
            x=ctx['coords'][:,0],y=ctx['coords'][:,1]))
        # Identity path adapter replaces only the historical drive rewrite.
        sensor = _load('teacher_sensor',patch_path_text=str)
        enriched = sensor.add_sensor_features(candidates,canonical,pd.DataFrame(rows))
        safe = json.loads(Path(__file__).with_name('teacher_protocol.json').read_text())
        p70 = deepcopy(safe['evidence']); p71 = safe['risk']
        # Both historical boundaries were CSV; round-trip new data, not old caches.
        sensor_path = output/'fresh_sensor_evidence.csv'
        enriched.to_csv(sensor_path,index=False)
        p70['candidate_input']['path'] = str(sensor_path)
        evidence = _load('teacher_evidence', B67=f.b67).prepare_candidate_evidence(p70)
        tf = training_functions()
        evidence = tf.teacher_generation.apply_relaxed_evidence_gate(evidence,p71)
        context = dict(ctx,dataset_id=sample['dataset'],sample_id=sample['dataset'])
        prepared,_ = tf.teacher_generation.prepare_dataset_candidates(evidence,context)
        accepted = []
        for row in rows:
            base_labels = sensor.load_prediction_map(pd.Series(row),ctx['spot_ids'])
            scored,_ = tf.teacher_generation.score_method_candidates(prepared,base_labels,context,p71)
            decisions,_,_ = tf.teacher_generation.accept_scored_candidates(scored,base_labels,context,row['method'],
                p71['budget']['primary_maximum_unique_modified_fraction'])
            if not decisions.empty:
                accepted.append(decisions)
        columns = ['dataset_id','method','proposal_type','member_spots','fragmentation_risk','erosion_risk','boundary_leakage_risk','evidence_score']
        teacher = pd.concat(accepted,ignore_index=True) if accepted else pd.DataFrame(columns=columns)
        target = output/'teacher_acceptance.csv'
        teacher.to_csv(target,index=False)
        (output/'teacher_metadata.json').write_text(json.dumps(dict(dataset=sample['dataset'],
            workflows=[r['method'] for r in rows],sensor_method_denominator=14,
            proposal_rows=len(candidates),sensor_rows_after_original_short_id_join=len(enriched),
            accepted_teacher_rows=len(teacher),reference_annotations_read=False,
            supervised_loso_predictions_used=False,inputs=inputs,
            teacher_sha256=sha(target),protocol_sha256=sha(Path(__file__).with_name('teacher_protocol.json'))),indent=2))
    return target


def generate(manifest, output, held_out=None, audit_access=False):
    import os
    from .config import new_output
    path, samples = selected_samples(manifest,held_out)
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0 before starting Python')
    if any(len(s.get('upstream',[])) != 14 for s in samples):
        raise ValueError('The original teacher requires exactly 14 upstream workflows per included sample')
    out = new_output(output)
    os.environ.update(NUMBA_CACHE_DIR=str(out/'numba_cache'),MPLCONFIGDIR=str(out/'mplconfig'),TEMP=str(out),TMP=str(out))
    log = None
    if audit_access:
        from .audit import install
        from .training import resolve
        allowed = [path,Path(__file__).with_name('teacher_protocol.json')]
        for s in samples:
            allowed += [resolve(path.parent,s[k]) for k in ['expression','coordinates','feature_expression'] if s.get(k)]
            allowed += [resolve(path.parent,m['labels']) for m in s['upstream']]
        log = install(out,allowed)
    generated = {s['dataset']:str(generate_one(s,path.parent,out/s['dataset'])) for s in samples}
    (out/'teacher_manifest.json').write_text(json.dumps(dict(held_out_dataset=held_out,included_datasets=list(generated),teachers=generated),indent=2))
    if log: log.flush()
    print(out/'teacher_manifest.json')
