from __future__ import annotations
import math, hashlib
from pathlib import Path
from collections import Counter, defaultdict, deque
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import io, json, time
import joblib
ROOT = Path('.')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            digest.update(chunk)
    return digest.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def feature_pcs(path, kind, spots):
    data = sc.read_10x_h5(path) if kind == '10x_h5' else ad.read_h5ad(path)
    data.obs_names = data.obs_names.astype(str)
    data.var_names_make_unique()
    if set(spots) - set(data.obs_names):
        raise RuntimeError('Missing expression barcodes')
    data = data[spots].copy()
    sc.pp.normalize_total(data, target_sum=10000.0)
    sc.pp.log1p(data)
    if data.n_vars > 3000:
        sc.pp.highly_variable_genes(data, n_top_genes=3000, flavor='seurat', subset=True)
    sc.pp.pca(data, n_comps=max(2, min(30, data.n_obs - 1, data.n_vars - 1)), zero_center=True, random_state=20260805)
    x = np.asarray(data.obsm['X_pca'], dtype=np.float32)
    med = np.median(x, axis=0)
    return ((x - med) / np.maximum(1.4826 * np.median(np.abs(x - med), axis=0), 0.001)).astype(np.float32)

def generate_sample(row, f):
    coords_frame = pd.read_csv(ROOT / row.coordinates, dtype={'spot_id': str})
    spots = coords_frame.spot_id.tolist()
    coords = coords_frame[['x', 'y']].to_numpy(float)
    folder = (ROOT / row.cfar_expression).parent
    if row.source == 'internal':
        (x, _, aligned) = f.expr.read_expression(folder, spots)
        if list(aligned) != spots:
            raise RuntimeError('CFAR barcode order differs')
        pcs = f.internal.pca_scores(x)
        scores = f.internal.local_scores(pcs, coords)
        adj = f.internal.adjacency(coords)
        rows = []
        seen = set()
        for (seed, spot) in enumerate(spots):
            rows.append(f.internal.make_row(row.dataset, row.sample, [spot], scores[seed], 'singleton', f'seed={seed}'))
            seen.add((spot,))
            for j in adj[seed]:
                if seed < j:
                    pair = tuple(sorted([spot, spots[j]]))
                    if pair not in seen:
                        rows.append(f.internal.make_row(row.dataset, row.sample, list(pair), (scores[seed] + scores[j]) / 2, 'edge_pair', f'edge={seed}-{j}'))
                        seen.add(pair)
            for size in [3, 4, 5, 8, 10]:
                indices = f.internal.grow(seed, size, scores, adj)
                if len(indices) != size:
                    continue
                members = tuple(sorted((spots[j] for j in indices)))
                if members in seen:
                    continue
                seen.add(members)
                rows.append(f.internal.make_row(row.dataset, row.sample, list(members), np.mean(scores[indices]) + scores[seed], f'core_{size}', f'seed={seed};target={size}'))
        candidates = pd.DataFrame(rows)
    else:
        (x, aligned) = f.ext.read_expression(folder, spots)
        if list(aligned) != spots:
            raise RuntimeError('CFAR barcode order differs')
        pcs = f.ext.pca_scores(x)
        scores = f.ext.local_scores(pcs, coords)
        (candidates, _) = f.ext.generate_candidates(row.dataset, spots, coords, scores)
    candidates = pd.read_csv(io.StringIO(candidates.to_csv(index=False)))
    candidates['score_per_spot'] = candidates.proposal_score / np.sqrt(candidates.n_spots.clip(lower=1))
    if row.source == 'internal':
        candidates = pd.read_csv(io.StringIO(candidates.to_csv(index=False)))
    candidates['canonical_member_spots'] = candidates.member_spots.map(f.i.canonical_spots)
    candidates['candidate_key'] = [f.i.candidate_key(d, p, m) for (d, p, m) in zip(candidates.dataset_id, candidates.proposal_type, candidates.canonical_member_spots)]
    candidates['candidate_row_key'] = candidates.dataset_id + '::row::' + candidates.groupby('dataset_id', sort=False).cumcount().astype(str)
    group = candidates.groupby(['dataset_id', 'proposal_type'], sort=False)
    candidates['proposal_percentile'] = group.proposal_score.transform(f.i.percentile_rank)
    candidates['score_per_spot_percentile'] = group.score_per_spot.transform(f.i.percentile_rank)
    candidates['compactness_raw'] = 1 / np.log1p(1 + candidates.n_spots.clip(lower=1))
    candidates['compactness_percentile'] = group.compactness_raw.transform(f.i.percentile_rank)
    candidates = f.i.add_multiscale_persistence(candidates)
    if row.source == 'external_second':
        adj = f.ext.radius_adjacency(coords)
    else:
        adj = f.b2.adjacency_at_radius(coords, f.b2.infer_radius(coords))
    ctx = {'spot_ids': spots, 'coords': coords, 'adjacency': adj}
    fp = feature_pcs(ROOT / row.expression, row.expression_kind, spots)
    (edges, anomaly) = f.c.edge_distances(fp, adj)
    index = {s: i for (i, s) in enumerate(spots)}
    indices = [{index[s] for s in f.b67.parse_spots(v)} for v in candidates.canonical_member_spots]
    expr = pd.DataFrame([f.c.candidate_features(ii, fp, adj, edges, anomaly) for ii in indices])
    enhanced = pd.concat([candidates.reset_index(drop=True), expr], axis=1)
    return (enhanced, ctx, indices, scores)

def build_pair_features(row, candidates, ctx, indices, f, run):
    started = time.time()
    d = row.dataset
    method = row.method
    basefile = pd.read_csv(ROOT / row.upstream, dtype=str).fillna('')
    barcode = row.barcode_col or ('spot_id' if 'spot_id' in basefile else basefile.columns[0])
    label = row.label_col
    if label not in basefile:
        raise RuntimeError('Explicit upstream label field missing:' + label)
    aligned = basefile.drop_duplicates(barcode).set_index(barcode)[label].reindex(ctx['spot_ids'])
    barcode_alignment = float(aligned.notna().mean())
    positional_fallback = barcode_alignment < 0.5 and len(basefile) == len(ctx['spot_ids'])
    base = f.b67.load_prediction(ROOT / row.upstream, pd.Index(ctx['spot_ids']), barcode, label)
    base.index = pd.Index(ctx['spot_ids'])
    frame = candidates.copy()
    codes = f.i.prediction_codes(base, ctx['spot_ids'])
    (boundary, rare, _, _) = f.helper.raw_sensor_evidence(frame, ctx, codes)
    frame['single_boundary_mean'] = boundary[:, 0]
    frame['single_rare_cluster_support'] = rare[:, 0]
    frame['single_island_support'] = (boundary[:, 0] >= 0.75).astype(int)
    frame['single_boundary_percentile'] = frame.groupby(['dataset_id', 'proposal_type'], sort=False).single_boundary_mean.transform(f.i.percentile_rank)
    (vals, counts) = np.unique(codes[:, 0], return_counts=True)
    frame = pd.concat([frame, pd.DataFrame([f.i.label_features(ii, codes[:, 0], dict(zip(map(int, vals), map(int, counts)))) for ii in indices])], axis=1)
    safety = f.helper.topology_safety(set(frame.candidate_row_key), frame, indices, base, ctx)
    cols = ['fragmentation_risk', 'cross_component_risk', 'boundary_leakage_risk', 'erosion_risk', 'safety_score', 'touched_upstream_components']
    frame['topology_valid'] = [key in safety for key in frame.candidate_row_key]
    for col in cols:
        frame[col] = [safety.get(key, {}).get(col, 1.0) for key in frame.candidate_row_key]
    labels = base.to_numpy(str)
    sizes = Counter(labels.tolist())
    percent = pd.Series(list(sizes.values()), index=list(sizes)).rank(method='average', pct=True).to_dict()
    summary = f.m.map_summary(labels, ctx['adjacency'])
    mf = pd.DataFrame([f.m.candidate_context(ii, labels, ctx['adjacency'], sizes, percent, summary) for ii in indices])
    frame = pd.concat([frame, mf], axis=1)
    frame['method'] = method
    return (frame, base)

def generate_pair(row, candidates, ctx, indices, f, run):
    started = time.time()
    d = row.dataset
    method = row.method
    basefile = pd.read_csv(ROOT / row.upstream, dtype=str).fillna('')
    barcode = row.barcode_col or ('spot_id' if 'spot_id' in basefile else basefile.columns[0])
    label = row.label_col
    if label not in basefile:
        raise RuntimeError('Explicit upstream label field missing:' + label)
    aligned = basefile.drop_duplicates(barcode).set_index(barcode)[label].reindex(ctx['spot_ids'])
    barcode_alignment = float(aligned.notna().mean())
    positional_fallback = barcode_alignment < 0.5 and len(basefile) == len(ctx['spot_ids'])
    base = f.b67.load_prediction(ROOT / row.upstream, pd.Index(ctx['spot_ids']), barcode, label)
    base.index = pd.Index(ctx['spot_ids'])
    frame = candidates.copy()
    codes = f.i.prediction_codes(base, ctx['spot_ids'])
    (boundary, rare, _, _) = f.helper.raw_sensor_evidence(frame, ctx, codes)
    frame['single_boundary_mean'] = boundary[:, 0]
    frame['single_rare_cluster_support'] = rare[:, 0]
    frame['single_island_support'] = (boundary[:, 0] >= 0.75).astype(int)
    frame['single_boundary_percentile'] = frame.groupby(['dataset_id', 'proposal_type'], sort=False).single_boundary_mean.transform(f.i.percentile_rank)
    (vals, counts) = np.unique(codes[:, 0], return_counts=True)
    frame = pd.concat([frame, pd.DataFrame([f.i.label_features(ii, codes[:, 0], dict(zip(map(int, vals), map(int, counts)))) for ii in indices])], axis=1)
    safety = f.helper.topology_safety(set(frame.candidate_row_key), frame, indices, base, ctx)
    cols = ['fragmentation_risk', 'cross_component_risk', 'boundary_leakage_risk', 'erosion_risk', 'safety_score', 'touched_upstream_components']
    frame['topology_valid'] = [key in safety for key in frame.candidate_row_key]
    for col in cols:
        frame[col] = [safety.get(key, {}).get(col, 1.0) for key in frame.candidate_row_key]
    labels = base.to_numpy(str)
    sizes = Counter(labels.tolist())
    percent = pd.Series(list(sizes.values()), index=list(sizes)).rank(method='average', pct=True).to_dict()
    summary = f.m.map_summary(labels, ctx['adjacency'])
    mf = pd.DataFrame([f.m.candidate_context(ii, labels, ctx['adjacency'], sizes, percent, summary) for ii in indices])
    frame = pd.concat([frame, mf], axis=1)
    frame['method'] = method
    modelpath = Path(row.ranker_model)
    model = joblib.load(modelpath)
    types = model['proposal_types']
    x = f.g.feature_frame(frame, types)
    if list(x.columns) != list(model['feature_columns']):
        raise RuntimeError('Model feature order mismatch')
    score = f.k.predict(frame, model, types)
    if row.source == 'internal':
        combined = 0.8 * np.clip(frame.score_per_spot_percentile.to_numpy(float), 0, 1) + 0.2 * np.clip(score.score_expected_gain_rank.to_numpy(float), 0, 1) - 0.025 * np.clip(frame.cross_component_risk.to_numpy(float), 0, 1)
    else:
        combined = 0.8 * frame.score_per_spot_percentile.to_numpy(float) + 0.2 * score.score_expected_gain_rank.to_numpy(float) - 0.025 * frame.cross_component_risk.to_numpy(float)
    target = run / d / method
    target.mkdir(parents=True)
    frame.to_parquet(target / 'features.parquet', index=False)
    pd.concat([frame[['candidate_key']], x], axis=1).to_parquet(target / 'model_feature_array.parquet', index=False)
    pd.concat([frame[['candidate_key']], score, pd.Series(combined, name='selection_score')], axis=1).to_parquet(target / 'ranking.parquet', index=False)
    status = {'dataset': d, 'method': method, 'source': row.source, 'candidates': len(frame), 'features': x.shape[1], 'barcode_alignment_fraction': barcode_alignment, 'positional_fallback_used': positional_fallback, 'ranker_model': str(Path(modelpath)), 'ranker_model_sha256': sha(modelpath), 'template_status': 'BLOCKED_MISSING_V12_LOSO_MODELS', 'runtime_seconds': time.time() - started}
    if row.source != 'internal':
        bundlepath = Path(row.selector_model)
        bundle = joblib.load(bundlepath)
        teacher = bundle['teacher'].predict_proba(f.t.blind_features(frame, bundle['proposal_types']))[:, 1]
        utility = f.signed.signed_score(frame, bundle['signed_utility_classifier'], bundle['signed_utility_regressor'], bundle['proposal_types'], bundle['harm_weight'])
        (templates, _) = f.e.add_utility_tail(frame, teacher, utility, ctx['spot_ids'], method, bundle['extra_utility_budget'])
        if templates.empty:
            templates = pd.DataFrame(columns=['dataset_id', 'stage', 'proposal_type', 'n_spots'])
        templates.to_csv(target / 'selector_slots_generated.csv', index=False)
        variant = 'expression_anchor_expected_0.200_crossrisk_0.025_locked'
        (_, selected, search) = f.sel.exact_token_assignment(frame, combined, combined, templates, ctx['spot_ids'], method, variant)
        budget = int(math.floor(0.97 * selected.n_spots.sum() + 1e-12))
        chosen = []
        used = 0
        for rr in selected.sort_values(['selection_score', 'candidate_key'], ascending=[False, True]).itertuples():
            if used + rr.n_spots <= budget:
                chosen.append(rr.Index)
                used += rr.n_spots
        retained = selected.loc[chosen].copy()
        retained.to_csv(target / 'selected_regions.csv', index=False)
        output = base.copy()
        for (number, rr) in enumerate(retained.itertuples(), 1):
            output.loc[list(f.b67.parse_spots(rr.member_spots))] = f'RadarST_locked_retention_{method}_{number}'
        pd.DataFrame({'spot_id': ctx['spot_ids'], 'pred_label': output.to_numpy()}).to_csv(target / 'final_labels.csv', index=False)
        status.update(template_status='GENERATED_FROM_FROZEN_V12_FULL_MODEL', slot_model_sha256=sha(bundlepath), **search)
    dump(target / 'status.json', status)
    print(d, method, status['template_status'], len(frame), flush=True)
