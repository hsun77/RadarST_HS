from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import datetime,timezone
import hashlib,math,os,time,json
import numpy as np
import pandas as pd
from scipy import stats,sparse
from sklearn.neighbors import NearestNeighbors
import gseapy as gp

SEED = 20260912
RANDOM_BASE_SEED = 20260805
N_BLOCK_PERM = 10000
N_BOOT = 2000
N_GSEA_PERM = 1000
N_RANDOM = 500
MIN_SET_SIZE = 15
MAX_SET_SIZE = 500
MIN_SECTION_GENE = 12
TOL = 1e-08
MIN_CORR_N = 10
CONTEXT_ORDER = ['TLS/lymphoid microdomain', 'Malignant epithelial/tumor', 'Tumor boundary', 'Immune/inflammatory context', 'Stromal/benign tissue context', 'unassigned']
CANCER_ORDER = ['breast cancer', 'kidney cancer', 'lung cancer', 'PDAC']

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def stable_seed(*parts: object, base: int=SEED) -> int:
    token = '||'.join(map(str, parts)).encode('utf-8')
    return (base + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2 ** 32 - 1)

def bh_fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    out[valid] = np.minimum(1.0, q[inverse])
    return out

def canonical_members(value: object) -> str:
    return ';'.join(sorted({x for x in str(value).split(';') if x and x != 'nan'}))

def compute_frontend(sample: dict, m56, m72) -> dict:
    expression = sample['expression']
    standardized = expression - np.nanmean(expression, axis=0, keepdims=True)
    standardized = standardized / np.maximum(np.nanstd(standardized, axis=0, keepdims=True), 0.001)
    (u, singular, vt) = np.linalg.svd(standardized, full_matrices=False)
    n_pc = min(20, u.shape[1])
    pcs = (u[:, :n_pc] * singular[:n_pc]).astype(np.float32)
    loadings = vt[:n_pc].astype(np.float32)
    if sample['source_kind'].startswith('56C'):
        formal_pcs = m56.pca_scores(expression)
        if not np.array_equal(pcs, formal_pcs):
            if not np.allclose(np.abs(pcs), np.abs(formal_pcs), atol=1e-07, rtol=1e-07):
                raise RuntimeError(f"PCA reproduction failed for {sample['dataset_id']}")
        difference = sample['coords'][:, None, :] - sample['coords'][None, :, :]
        distance = np.sqrt(np.sum(difference * difference, axis=2))
        order = np.argsort(distance, axis=1)
        formal_score = m56.local_scores(pcs, sample['coords'])
    else:
        formal_pcs = m72.pca_scores(expression)
        if not np.array_equal(pcs, formal_pcs):
            if not np.allclose(np.abs(pcs), np.abs(formal_pcs), atol=1e-07, rtol=1e-07):
                raise RuntimeError(f"PCA reproduction failed for {sample['dataset_id']}")
        order = NearestNeighbors(n_neighbors=min(40, len(sample['coords']))).fit(sample['coords']).kneighbors(sample['coords'], return_distance=False)
        formal_score = m72.local_scores(pcs, sample['coords'])
    n_spots = len(order)
    signed = np.zeros((n_spots, n_pc), dtype=np.float32)
    backgrounds: list[np.ndarray] = []
    for (i, row) in enumerate(order):
        bg = row[8:40] if len(row) >= 40 else row[2:]
        backgrounds.append(np.asarray(bg, dtype=np.int32))
        if len(bg) < 4:
            continue
        median = np.median(pcs[bg], axis=0)
        mad = np.maximum(np.median(np.abs(pcs[bg] - median), axis=0), 0.2)
        signed[i] = ((pcs[i] - median) / mad).astype(np.float32)
    absolute = np.abs(signed)
    k = min(8, n_pc)
    top_idx = np.argsort(absolute, axis=1, kind='mergesort')[:, -k:]
    top_mask = np.zeros_like(absolute, dtype=bool)
    np.put_along_axis(top_mask, top_idx, True, axis=1)
    recomputed_score = np.take_along_axis(absolute, top_idx, axis=1).mean(axis=1)
    if not np.allclose(recomputed_score, formal_score, atol=1e-07, rtol=0):
        raise RuntimeError(f"Local-score implementation mismatch for {sample['dataset_id']}")
    selected_signed = np.where(top_mask, signed, 0.0).astype(np.float32)
    gene_signed = selected_signed @ loadings
    gene_magnitude = np.abs(selected_signed) @ np.abs(loadings)
    denom = np.maximum(np.sum(np.abs(selected_signed), axis=1, keepdims=True), 1e-12)
    gene_normalized = selected_signed / denom @ loadings
    contribution = np.abs(selected_signed) / denom
    return {**sample, 'standardized': standardized.astype(np.float32), 'pcs': pcs, 'loadings': loadings, 'signed': signed, 'absolute': absolute, 'top_idx': top_idx.astype(np.int16), 'top_mask': top_mask, 'cfar': recomputed_score.astype(np.float32), 'formal_function_score': np.asarray(formal_score, dtype=np.float64), 'contribution': contribution.astype(np.float32), 'gene_signed': gene_signed.astype(np.float32), 'gene_magnitude': gene_magnitude.astype(np.float32), 'gene_normalized': gene_normalized.astype(np.float32), 'backgrounds': backgrounds}

def vectorized_row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = stats.rankdata(a, axis=1, method='average').astype(np.float64)
    rb = stats.rankdata(b, axis=1, method='average').astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(ra * ra, axis=1) * np.sum(rb * rb, axis=1))
    return np.divide(np.sum(ra * rb, axis=1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)

def save_sample_npz(result: dict) -> None:
    max_bg = max((len(x) for x in result['backgrounds']), default=0)
    bg = np.full((len(result['backgrounds']), max_bg), -1, dtype=np.int32)
    bg_n = np.zeros(len(result['backgrounds']), dtype=np.int16)
    for (i, values) in enumerate(result['backgrounds']):
        bg[i, :len(values)] = values
        bg_n[i] = len(values)
    np.savez_compressed(PROJECT / 'intermediate/samples' / f"{result['dataset_id']}.npz", spot_ids=result['spot_ids'], genes=result['genes'], coords=result['coords'], expression=result['expression'], standardized=result['standardized'], pcs=result['pcs'], loadings=result['loadings'], signed=result['signed'], top_idx=result['top_idx'], cfar=result['cfar'], contribution=result['contribution'], gene_signed=result['gene_signed'], gene_magnitude=result['gene_magnitude'], gene_normalized=result['gene_normalized'], fidelity_rho=result['fidelity_rho'], backgrounds=bg, background_n=bg_n)

def save_candidate_npz(dataset_id: str, group: pd.DataFrame, result: dict, source_pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    spot_lookup = {spot: i for (i, spot) in enumerate(result['spot_ids'])}
    source_pool = source_pool.copy()
    source_pool['candidate_member_key'] = source_pool['member_spots'].map(canonical_members)
    source_pool['source_key'] = source_pool['proposal_type'].astype(str) + '||' + source_pool['candidate_member_key']
    if source_pool['source_key'].duplicated().any():
        source_pool = source_pool.sort_values('candidate_id', kind='mergesort').drop_duplicates('source_key')
    source_index = source_pool.set_index('source_key')
    rows = []
    (signed, magnitude, normalized, pc_share) = ([], [], [], [])
    missing_source = 0
    for row in group.itertuples(index=False):
        member_ids = [x for x in str(row.member_spots).split(';') if x]
        if any((x not in spot_lookup for x in member_ids)):
            raise RuntimeError(f'Unaligned retained member in {dataset_id}: {row.candidate_key}')
        idx = np.asarray([spot_lookup[x] for x in member_ids], dtype=int)
        signed.append(np.median(result['gene_signed'][idx], axis=0))
        magnitude.append(np.median(result['gene_magnitude'][idx], axis=0))
        normalized.append(np.median(result['gene_normalized'][idx], axis=0))
        pc_share.append(np.mean(result['contribution'][idx], axis=0))
        key = f'{row.proposal_type}||{canonical_members(row.member_spots)}'
        src = source_index.loc[key] if key in source_index.index else None
        if src is None:
            missing_source += 1
            proposal_score = np.nan
            source_candidate_id = ''
        else:
            proposal_score = float(src['proposal_score'])
            source_candidate_id = str(src['candidate_id'])
        rows.append({'sample': dataset_id, 'candidate_key': row.candidate_key, 'candidate_id': row.candidate_id, 'source_candidate_id': source_candidate_id, 'proposal_type': row.proposal_type, 'member_spots': row.member_spots, 'background_spots': row.background_spots, 'n_background': int(row.n_background) if str(row.n_background) else len([x for x in str(row.background_spots).split(';') if x]), 'candidate_size': int(row.candidate_size), 'cohort': row.cohort, 'cancer_type': row.cancer_type, 'biological_context': row.biological_context, 'workflow_recurrence_count': int(row.method_recurrence), 'formal_workflows_supporting_region': row.methods, 'candidate_molecular_score': float(row.candidate_molecular_score) if np.isfinite(row.candidate_molecular_score) else np.nan, 'formal_proposal_score': proposal_score, 'mean_spot_cfar': float(np.mean(result['cfar'][idx])), 'median_spot_cfar': float(np.median(result['cfar'][idx])), 'max_spot_cfar': float(np.max(result['cfar'][idx])), 'median_member_spot_attribution_fidelity': float(np.nanmedian(result['fidelity_rho'][idx])), 'section_median_spot_attribution_fidelity': float(np.nanmedian(result['fidelity_rho'])), 'gene_index_id': dataset_id})
    signed_m = np.asarray(signed, dtype=np.float32)
    magnitude_m = np.asarray(magnitude, dtype=np.float32)
    normalized_m = np.asarray(normalized, dtype=np.float32)
    pc_share_m = np.asarray(pc_share, dtype=np.float32)
    np.savez_compressed(PROJECT / 'intermediate/candidates' / f'{dataset_id}.npz', candidate_key=group['candidate_key'].astype(str).to_numpy(dtype='U'), genes=np.asarray(result['genes'], dtype='U'), signed=signed_m, magnitude=magnitude_m, normalized=normalized_m, pc_share=pc_share_m)
    return (pd.DataFrame(rows), {'missing_source_candidates': missing_source, 'n_retained': len(group)})

def trim_mean(values: np.ndarray, proportion: float=0.1) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(stats.trim_mean(values, proportion))

def load_candidate_payload(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(PROJECT / 'intermediate/candidates' / f'{dataset_id}.npz', allow_pickle=True)
    return (data['candidate_key'], data['genes'], data['signed'], data['magnitude'], data['pc_share'])

def section_gene_and_landscape(candidate_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_rows = []
    global_signed: dict[str, list[np.ndarray]] = defaultdict(list)
    global_magnitude: dict[str, list[np.ndarray]] = defaultdict(list)
    for dataset_id in candidate_meta['sample'].drop_duplicates():
        (_, genes, signed, magnitude, _) = load_candidate_payload(dataset_id)
        for (j, gene_raw) in enumerate(genes.astype(str)):
            gene = gene_raw.upper()
            values = signed[:, j].astype(float)
            mags = magnitude[:, j].astype(float)
            global_signed[gene].append(values)
            global_magnitude[gene].append(mags)
            section_rows.append({'sample': dataset_id, 'gene': gene, 'n_regions': len(values), 'section_median_attribution': float(np.median(values)), 'section_mean_attribution': float(np.mean(values)), 'section_trimmed_mean_attribution': trim_mean(values), 'section_fraction_positive': float(np.mean(values > 0)), 'section_median_magnitude': float(np.median(mags))})
    section = pd.DataFrame(section_rows)
    section.to_csv(PROJECT / 'intermediate/section_gene_cfar_attribution.csv.gz', index=False, compression='gzip')
    section_pivot = section.pivot(index='sample', columns='gene', values='section_median_attribution')
    eligible = section.groupby('gene')['sample'].nunique()
    eligible = eligible[eligible >= MIN_SECTION_GENE].index
    section_pivot = section_pivot.reindex(index=candidate_meta['sample'].drop_duplicates(), columns=eligible)
    matrix = section_pivot.to_numpy(float)
    observed = np.nanmean(matrix, axis=0)
    rng = np.random.default_rng(stable_seed('gene_section_signflip'))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_BLOCK_PERM, matrix.shape[0]), replace=True)
    n_available = np.sum(np.isfinite(matrix), axis=0)
    exceed = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, matrix.shape[1], 512):
        stop = min(start + 512, matrix.shape[1])
        block = np.nan_to_num(matrix[:, start:stop], nan=0.0)
        permuted = signs @ block / np.maximum(1, n_available[start:stop])[None, :]
        exceed[start:stop] = np.sum(np.abs(permuted) >= np.abs(observed[start:stop])[None, :], axis=0)
    gene_p = (1 + exceed) / (N_BLOCK_PERM + 1)
    gene_fdr = bh_fdr(gene_p)
    sign_fraction = np.maximum(np.nanmean(matrix > 0, axis=0), np.nanmean(matrix < 0, axis=0))
    abs_matrix = np.abs(matrix)
    dominance = np.nanmax(abs_matrix, axis=0) / np.maximum(np.nansum(abs_matrix, axis=0), 1e-12)
    inference = pd.DataFrame({'gene': eligible, 'section_equal_mean_attribution': observed, 'section_signflip_p': gene_p, 'section_signflip_fdr': gene_fdr, 'fraction_sections_same_direction': sign_fraction, 'maximum_section_absolute_share': dominance}).set_index('gene')
    rows = []
    for gene in sorted(global_signed):
        values = np.concatenate(global_signed[gene]).astype(float)
        mags = np.concatenate(global_magnitude[gene]).astype(float)
        sec = section[section['gene'].eq(gene)]
        record = {'gene': gene, 'region_weighted_median_attribution': float(np.median(values)), 'region_weighted_mean_attribution': float(np.mean(values)), 'region_weighted_trimmed_mean_attribution': trim_mean(values), 'fraction_attribution_positive': float(np.mean(values > 0)), 'region_weighted_median_magnitude': float(np.median(mags)), 'section_equal_mean_attribution': float(sec['section_median_attribution'].mean()), 'section_median_attribution': float(sec['section_median_attribution'].median()), 'fraction_sections_positive': float((sec['section_median_attribution'] > 0).mean()), 'section_equal_mean_magnitude': float(sec['section_median_magnitude'].mean()), 'n_regions': len(values), 'n_sections': sec['sample'].nunique(), 'eligible_for_pathway': gene in inference.index}
        if gene in inference.index:
            record.update(inference.loc[gene].to_dict())
            if record['section_signflip_fdr'] < 0.05 and record['fraction_sections_same_direction'] >= 0.75:
                record['attribution_breadth'] = 'broad'
            elif record['maximum_section_absolute_share'] >= 0.5:
                record['attribution_breadth'] = 'section-driven'
            else:
                record['attribution_breadth'] = 'heterogeneous'
        else:
            record.update({'section_signflip_p': np.nan, 'section_signflip_fdr': np.nan, 'fraction_sections_same_direction': np.nan, 'maximum_section_absolute_share': np.nan, 'attribution_breadth': 'insufficient-section-coverage'})
        rows.append(record)
    landscape = pd.DataFrame(rows).sort_values(['section_equal_mean_magnitude', 'region_weighted_median_magnitude', 'gene'], ascending=[False, False, True], kind='mergesort').reset_index(drop=True)
    landscape.to_csv(PROJECT / 'cfar_gene_landscape.csv', index=False)
    return (section, landscape)

def pc_landscape(candidate_meta: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset_id, meta) in candidate_meta.groupby('sample', sort=False):
        (keys, genes, _, _, pc_share) = load_candidate_payload(dataset_id)
        if list(meta['candidate_key'].astype(str)) != list(keys.astype(str)):
            raise RuntimeError(f'Candidate row order mismatch for {dataset_id}')
        for (local_i, row) in enumerate(meta.itertuples(index=False)):
            for pc in range(pc_share.shape[1]):
                records.append({'sample': dataset_id, 'candidate_key': row.candidate_key, 'cancer_type': row.cancer_type, 'biological_context': row.biological_context, 'pc': pc + 1, 'contribution_fraction': float(pc_share[local_i, pc]), 'is_driver': bool(pc_share[local_i, pc] > 0)})
    long = pd.DataFrame(records)
    for (group_col, filename) in [('sample', 'pc_usage_by_section.csv'), ('cancer_type', 'pc_usage_by_cancer.csv'), ('biological_context', 'pc_usage_by_context.csv')]:
        summary = long.groupby([group_col, 'pc'], as_index=False).agg(n_candidates=('candidate_key', 'nunique'), driver_frequency=('is_driver', 'mean'), mean_contribution_fraction=('contribution_fraction', 'mean'), median_contribution_fraction=('contribution_fraction', 'median'))
        summary.to_csv(PROJECT / filename, index=False)
    long.to_parquet(PROJECT / 'intermediate/candidate_pc_contribution.parquet', index=False, compression='zstd')
    return long

def pc_gene_drivers(pc_long: pd.DataFrame) -> pd.DataFrame:
    usage = pc_long.groupby('pc', as_index=False).agg(global_driver_frequency=('is_driver', 'mean'), global_mean_contribution_fraction=('contribution_fraction', 'mean'), sections_represented=('sample', 'nunique'), cancer_types_represented=('cancer_type', 'nunique')).set_index('pc')
    rows = []
    for dataset_id in pc_long['sample'].drop_duplicates():
        data = np.load(PROJECT / 'intermediate/samples' / f'{dataset_id}.npz', allow_pickle=False)
        genes = data['genes'].astype(str)
        loadings = data['loadings']
        cancer = SAMPLE_SPECS[dataset_id][1]
        for pc in range(loadings.shape[0]):
            positive = np.argsort(loadings[pc], kind='mergesort')[-10:][::-1]
            negative = np.argsort(loadings[pc], kind='mergesort')[:10]
            for (direction, indices) in [('positive', positive), ('negative', negative)]:
                for (rank, j) in enumerate(indices, start=1):
                    rows.append({'sample': dataset_id, 'cancer_type': cancer, 'pc': pc + 1, 'gene': genes[j], 'loading': float(loadings[pc, j]), 'loading_direction': direction, 'loading_rank_within_direction': rank, **usage.loc[pc + 1].to_dict()})
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / 'pc_gene_driver_table.csv', index=False)
    return result

def select_representative(candidate_meta: pd.DataFrame, fidelity: pd.DataFrame) -> dict:
    eligible = candidate_meta[candidate_meta['candidate_size'].between(3, 5) & candidate_meta['biological_context'].ne('unassigned') & candidate_meta['formal_proposal_score'].notna()].copy()
    eligible = eligible[eligible['median_member_spot_attribution_fidelity'] >= eligible['section_median_spot_attribution_fidelity']]
    if eligible.empty:
        raise RuntimeError('No candidate satisfies the locked representative-case rule')
    cohort_median = eligible.groupby('cohort')['median_spot_cfar'].transform('median')
    eligible['distance_to_cohort_median_cfar'] = (eligible['median_spot_cfar'] - cohort_median).abs()
    selected = eligible.sort_values(['distance_to_cohort_median_cfar', 'sample', 'candidate_key'], kind='mergesort').iloc[0]
    lock = {'locked_utc': utc_now(), 'candidate_key': selected['candidate_key'], 'sample': selected['sample'], 'candidate_size': int(selected['candidate_size']), 'biological_context': selected['biological_context'], 'median_spot_cfar': float(selected['median_spot_cfar']), 'selection_rule': 'size 3-5; assigned frozen context; complete formal proposal/background; candidate median member-spot fidelity at or above its section spot-level median; closest CFAR intensity to cohort median; lexicographic tie break', 'pathway_results_read_during_selection': False}
    (PROJECT / 'REPRESENTATIVE_CASE_LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    return lock

def candidate_explanations(candidate_meta: pd.DataFrame) -> None:
    output = PROJECT / 'candidate_cfar_explanations.jsonl'
    with output.open('w', encoding='utf-8') as handle:
        for (dataset_id, meta) in candidate_meta.groupby('sample', sort=False):
            (keys, genes, signed, _, pc_share) = load_candidate_payload(dataset_id)
            for (i, row) in enumerate(meta.itertuples(index=False)):
                pc_order = np.argsort(pc_share[i], kind='mergesort')[::-1][:3]
                gene_order = np.argsort(np.abs(signed[i]), kind='mergesort')[::-1][:20]
                positive = np.argsort(signed[i], kind='mergesort')[::-1][:10]
                negative = np.argsort(signed[i], kind='mergesort')[:10]
                record = {'sample': dataset_id, 'candidate_id': row.candidate_id, 'candidate_key': row.candidate_key, 'candidate_size': int(row.candidate_size), 'mean_CFAR': float(row.mean_spot_cfar), 'median_CFAR': float(row.median_spot_cfar), 'top_PCs': [{'pc': int(j + 1), 'share': float(pc_share[i, j])} for j in pc_order], 'top_genes': [{'gene': str(genes[j]), 'attribution': float(signed[i, j])} for j in gene_order], 'top_positive_gene_attributions': [{'gene': str(genes[j]), 'attribution': float(signed[i, j])} for j in positive], 'top_negative_gene_attributions': [{'gene': str(genes[j]), 'attribution': float(signed[i, j])} for j in negative]}
                handle.write(json.dumps(record) + '\n')

def parse_gmt(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                result[parts[0]] = {x.strip().upper() for x in parts[2:] if x.strip()}
    return result

def deterministic_rank(frame: pd.DataFrame, score: str, secondary: str) -> pd.DataFrame:
    rank = frame[['gene', score, secondary]].dropna().sort_values([score, secondary, 'gene'], ascending=[False, False, True], kind='mergesort')[['gene', score]].rename(columns={score: 'score'})
    scale = max(1.0, float(rank['score'].abs().max()))
    for (_, positions) in rank.groupby('score', sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) > 1:
            rank.iloc[positions, rank.columns.get_loc('score')] += np.linspace(scale * 1e-12, -scale * 1e-12, len(positions))
    return rank

def normalize_gsea(frame: pd.DataFrame, library: str, ranking: str, gene_sets: dict[str, set[str]], universe: set[str]) -> pd.DataFrame:
    rename = {'Term': 'term', 'ES': 'es', 'NES': 'nes', 'NOM p-val': 'nominal_p', 'FDR q-val': 'fdr', 'FWER p-val': 'fwer_p', 'Tag %': 'tag_fraction', 'Gene %': 'gene_fraction', 'Lead_genes': 'leading_edge_genes'}
    out = frame.rename(columns=rename).copy()
    fields = ['term', 'es', 'nes', 'nominal_p', 'fdr', 'fwer_p', 'tag_fraction', 'gene_fraction', 'leading_edge_genes']
    for field in fields:
        if field not in out:
            out[field] = np.nan
    out = out[fields]
    for field in ['es', 'nes', 'nominal_p', 'fdr', 'fwer_p']:
        out[field] = pd.to_numeric(out[field], errors='coerce')
    out.insert(0, 'ranking', ranking)
    out.insert(0, 'library', library)
    out['mapped_gene_set_size'] = out['term'].map(lambda term: len(gene_sets.get(str(term), set()) & universe))
    out['gsea_permutations'] = N_GSEA_PERM
    out['independent_unit_note'] = 'aggregate descriptive preranked GSEA; section-block sign flips define section-aware uncertainty'
    return out

def run_gsea(rank: pd.DataFrame, library: str, gene_sets: dict[str, set[str]], ranking: str) -> pd.DataFrame:
    result = gp.prerank(rnk=rank[['gene', 'score']], gene_sets=str(GENESETS[library]), min_size=MIN_SET_SIZE, max_size=MAX_SET_SIZE, permutation_num=N_GSEA_PERM, weight=1.0, ascending=False, threads=max(1, min(8, os.cpu_count() or 1)), seed=SEED, outdir=None, no_plot=True, verbose=False)
    return normalize_gsea(result.res2d, library, ranking, gene_sets, set(rank['gene']))

def block_permutation(library: str, gene_sets: dict[str, set[str]], genes: pd.Index, section_effect: np.ndarray) -> pd.DataFrame:
    ranks = np.full_like(section_effect, np.nan, dtype=np.float64)
    for i in range(section_effect.shape[0]):
        valid = np.isfinite(section_effect[i])
        if valid.any():
            values = stats.rankdata(section_effect[i, valid], method='average')
            ranks[i, valid] = values / (len(values) + 1.0) - 0.5
    gene_to_i = {gene: i for (i, gene) in enumerate(genes)}
    (terms, contrasts) = ([], [])
    for (term, members) in gene_sets.items():
        idx = np.asarray([gene_to_i[g] for g in members if g in gene_to_i], dtype=int)
        if not MIN_SET_SIZE <= len(idx) <= MAX_SET_SIZE:
            continue
        mask = np.ones(len(genes), dtype=bool)
        mask[idx] = False
        contrast = np.nanmean(ranks[:, idx], axis=1) - np.nanmean(ranks[:, mask], axis=1)
        terms.append((term, len(idx)))
        contrasts.append(contrast)
    contrast_matrix = np.column_stack(contrasts)
    observed = np.nanmean(contrast_matrix, axis=0)
    available = np.sum(np.isfinite(contrast_matrix), axis=0)
    rng = np.random.default_rng(stable_seed(library, 'section_block'))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_BLOCK_PERM, section_effect.shape[0]), replace=True)
    exceed = np.zeros(len(terms), dtype=np.int64)
    for start in range(0, len(terms), 512):
        stop = min(start + 512, len(terms))
        permuted = signs @ np.nan_to_num(contrast_matrix[:, start:stop], nan=0.0) / np.maximum(1, available[start:stop])[None, :]
        exceed[start:stop] = np.sum(np.abs(permuted) >= np.abs(observed[start:stop])[None, :], axis=0)
    p = (1 + exceed) / (N_BLOCK_PERM + 1)
    result = pd.DataFrame({'library': library, 'term': [x[0] for x in terms], 'mapped_gene_set_size': [x[1] for x in terms], 'block_es_observed': observed, 'empirical_p': p, 'n_section_block_permutations': N_BLOCK_PERM, 'n_sections': section_effect.shape[0], 'n_sections_with_mapped_effect': available, 'statistic_definition': 'mean centered within-section gene rank in set minus nonset, equally averaged across sections'})
    result['fdr'] = bh_fdr(result['empirical_p'])
    return result.sort_values(['fdr', 'empirical_p', 'term'], kind='mergesort').reset_index(drop=True)

def aggregate_pathways(section: pd.DataFrame, landscape: pd.DataFrame) -> tuple[dict[str, dict[str, set[str]]], pd.DataFrame]:
    eligible = landscape[landscape['eligible_for_pathway']].copy()
    genes = pd.Index(eligible['gene'].astype(str))
    section_ids = pd.Index(section['sample'].drop_duplicates())
    section_matrix = section.pivot(index='sample', columns='gene', values='section_median_attribution').reindex(index=section_ids, columns=genes).to_numpy(float)
    region_rank = deterministic_rank(eligible, 'region_weighted_median_attribution', 'region_weighted_mean_attribution')
    section_rank = deterministic_rank(eligible, 'section_equal_mean_attribution', 'section_median_attribution')
    libraries = {name: parse_gmt(path) for (name, path) in GENESETS.items()}
    merged = []
    for (library, gene_sets) in libraries.items():
        started = time.time()
        region = run_gsea(region_rank, library, gene_sets, 'region_weighted')
        section_equal = run_gsea(section_rank, library, gene_sets, 'section_equal')
        block = block_permutation(library, gene_sets, genes, section_matrix)
        region.to_csv(PROJECT / f'{library}_cfar_regionweighted.csv', index=False)
        section_equal.to_csv(PROJECT / f'{library}_cfar_sectionequal.csv', index=False)
        block.to_csv(PROJECT / f'{library}_cfar_blockperm.csv', index=False)
        r = region.add_prefix('region_').rename(columns={'region_library': 'library', 'region_term': 'term'})
        s = section_equal.add_prefix('section_').rename(columns={'section_library': 'library', 'section_term': 'term'})
        b = block.add_prefix('block_').rename(columns={'block_library': 'library', 'block_term': 'term'})
        merged.append(r.merge(s, on=['library', 'term'], how='outer').merge(b, on=['library', 'term'], how='outer'))
        print(f'[pathway] {library}: region={len(region)} block={len(block)} seconds={time.time() - started:.1f}', flush=True)
    master = pd.concat(merged, ignore_index=True)
    master['region_section_same_direction'] = np.sign(master['region_nes']) == np.sign(master['section_nes'])
    master['section_block_same_direction'] = np.sign(master['section_nes']) == np.sign(master['block_block_es_observed'])
    master['pathway_class'] = 'E NO SUPPORT'
    heterogeneous = master['region_fdr'].fillna(1).lt(0.05)
    broad = heterogeneous & master['region_section_same_direction'] & master['block_fdr'].fillna(1).lt(0.05) & master['section_block_same_direction']
    master.loc[heterogeneous, 'pathway_class'] = 'B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM'
    master.loc[broad, 'pathway_class'] = 'A BROAD CFAR-ATTRIBUTED PROGRAM'
    master.to_csv(PROJECT / 'intermediate/cfar_pathway_aggregate_master.csv', index=False)
    return (libraries, master)

def select_followup_terms(master: pd.DataFrame, libraries: dict[str, dict[str, set[str]]], eligible_genes: set[str]) -> pd.DataFrame:
    rows = []
    mandatory_tokens = ['MTORC1_SIGNALING', 'P53_PATHWAY', 'B_CELL_RECEPTOR_SIGNALING']
    for library in ['hallmark', 'reactome', 'gobp']:
        subset = master[master['library'].eq(library)].copy()
        subset['block_fdr_sort'] = subset['block_fdr'].fillna(1)
        subset['region_fdr_sort'] = subset['region_fdr'].fillna(1)
        subset['abs_section_nes'] = subset['section_nes'].abs().fillna(0)
        subset = subset.sort_values(['block_fdr_sort', 'region_fdr_sort', 'abs_section_nes', 'term'], ascending=[True, True, False, True], kind='mergesort')
        limit = len(subset) if library == 'hallmark' else 20
        selected = set(subset.head(limit)['term'].astype(str))
        for term in subset['term'].astype(str):
            if any((token in term.upper() for token in mandatory_tokens)):
                selected.add(term)
        for term in sorted(selected):
            mapped = libraries[library].get(term, set()) & eligible_genes
            if len(mapped) < MIN_SET_SIZE:
                continue
            row = subset[subset['term'].eq(term)].iloc[0]
            rows.append({'library': library, 'term': term, 'mapped_gene_set_size_global': len(mapped), 'selection_reason': 'all Hallmark' if library == 'hallmark' else 'locked named term' if any((token in term.upper() for token in mandatory_tokens)) else 'top-20 deterministic aggregate evidence ordering', 'aggregate_pathway_class': row['pathway_class'], 'region_nes': row['region_nes'], 'region_fdr': row['region_fdr'], 'section_nes': row['section_nes'], 'block_fdr': row['block_fdr']})
    selected = pd.DataFrame(rows).sort_values(['library', 'term'], kind='mergesort').reset_index(drop=True)
    selected['pathway_index'] = np.arange(len(selected), dtype=int)
    selected.to_csv(PROJECT / 'intermediate/cfar_candidate_pathway_lock.csv', index=False)
    return selected

def centered_row_ranks(matrix: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(matrix, axis=1, method='average').astype(np.float32)
    return ranks / (matrix.shape[1] + 1.0) - 0.5

def pathway_membership(genes: np.ndarray, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {gene.upper(): i for (i, gene) in enumerate(genes.astype(str))}
    membership = np.zeros((len(genes), len(selected)), dtype=np.float32)
    sizes = np.zeros(len(selected), dtype=np.int32)
    for (j, row) in enumerate(selected.itertuples(index=False)):
        idx = [lookup[g] for g in libraries[row.library].get(row.term, set()) if g in lookup]
        sizes[j] = len(idx)
        if idx:
            membership[idx, j] = 1.0
    return (membership, sizes)

def set_contrast_from_ranks(ranks: np.ndarray, membership: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    set_sum = ranks @ membership
    total = ranks.sum(axis=1, keepdims=True)
    n_gene = ranks.shape[1]
    set_mean = np.divide(set_sum, sizes[None, :], out=np.full_like(set_sum, np.nan), where=sizes[None, :] > 0)
    non_n = n_gene - sizes
    non_mean = np.divide(total - set_sum, non_n[None, :], out=np.full_like(set_sum, np.nan), where=non_n[None, :] > 0)
    result = set_mean - non_mean
    result[:, sizes < 5] = np.nan
    return result.astype(np.float32)

def mean_membership_matrix(texts: pd.Series, spot_lookup: dict[str, int], n_spots: int) -> sparse.csr_matrix:
    (rows, cols, values) = ([], [], [])
    for (i, text) in enumerate(texts.astype(str)):
        idx = sorted({spot_lookup[x] for x in text.split(';') if x in spot_lookup})
        if not idx:
            continue
        rows.extend([i] * len(idx))
        cols.extend(idx)
        values.extend([1.0 / len(idx)] * len(idx))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), n_spots), dtype=np.float32)

def symmetric_knn_graph(coords: np.ndarray, k: int=6) -> dict[int, set[int]]:
    graph = {i: set() for i in range(len(coords))}
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(return_distance=False)
    for (i, row) in enumerate(neighbors):
        for j in row:
            j = int(j)
            if i != j:
                graph[i].add(j)
                graph[j].add(i)
    return graph

def graph_shells(graph: dict[int, set[int]], core: list[int]) -> tuple[list[int], list[int]]:
    core_set = set(core)
    ring1 = set().union(*(graph[i] for i in core_set)) - core_set if core_set else set()
    ring2 = set().union(*(graph[i] for i in ring1)) - core_set - ring1 if ring1 else set()
    return (sorted(ring1), sorted(ring2))

def section_bootstrap_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(stable_seed('section_bootstrap', key))
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))

def candidate_pathway_stage(candidate_meta: pd.DataFrame, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (correlation_rows, context_payload, gradient_rows) = ([], [], [])
    all_candidate_rows = []
    for (dataset_id, meta) in candidate_meta.groupby('sample', sort=False):
        data = np.load(PROJECT / 'intermediate/samples' / f'{dataset_id}.npz', allow_pickle=False)
        cand = np.load(PROJECT / 'intermediate/candidates' / f'{dataset_id}.npz', allow_pickle=True)
        genes = data['genes'].astype(str)
        (membership, sizes) = pathway_membership(genes, selected, libraries)
        candidate_rank = centered_row_ranks(cand['signed'])
        candidate_pathway = set_contrast_from_ranks(candidate_rank, membership, sizes)
        spot_rank = centered_row_ranks(data['expression'])
        spot_direct_pathway = set_contrast_from_ranks(spot_rank, membership, sizes)
        spot_attr_rank = centered_row_ranks(data['gene_signed'])
        spot_attr_pathway = set_contrast_from_ranks(spot_attr_rank, membership, sizes)
        spot_lookup = {spot: i for (i, spot) in enumerate(data['spot_ids'].astype(str))}
        core_matrix = mean_membership_matrix(meta['member_spots'], spot_lookup, len(spot_lookup))
        bg_matrix = mean_membership_matrix(meta['background_spots'], spot_lookup, len(spot_lookup))
        direct_local = np.asarray(core_matrix @ spot_direct_pathway - bg_matrix @ spot_direct_pathway, dtype=np.float32)
        if list(meta['candidate_key'].astype(str)) != list(cand['candidate_key'].astype(str)):
            raise RuntimeError(f'Candidate payload order mismatch for pathway stage: {dataset_id}')
        candidate_quantile = pd.qcut(meta['median_spot_cfar'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str).to_numpy()
        np.savez_compressed(PROJECT / 'intermediate/pathways' / f'{dataset_id}.npz', candidate_key=cand['candidate_key'], terms=selected['term'].astype(str).to_numpy(), libraries=selected['library'].astype(str).to_numpy(), candidate_cfar_pathway=candidate_pathway, direct_local_pathway=direct_local)
        for (j, pathway) in enumerate(selected.itertuples(index=False)):
            valid = np.isfinite(direct_local[:, j]) & np.isfinite(meta['median_spot_cfar'].to_numpy(float))
            (rho, p) = stats.spearmanr(meta['median_spot_cfar'].to_numpy(float)[valid], direct_local[valid, j]) if valid.sum() >= MIN_CORR_N else (np.nan, np.nan)
            correlation_rows.append({'sample': dataset_id, 'library': pathway.library, 'term': pathway.term, 'n_candidates': int(valid.sum()), 'spearman_rho': float(rho) if np.isfinite(rho) else np.nan, 'descriptive_p': float(p) if np.isfinite(p) else np.nan, 'minimum_candidates_required': MIN_CORR_N, 'cfar_strength_definition': 'median member-spot formal CFAR score', 'pathway_contrast_definition': 'candidate minus frozen 30-spot local background centered expression-rank pathway score'})
            for (i, row) in enumerate(meta.itertuples(index=False)):
                context_payload.append({'sample': dataset_id, 'cohort': row.cohort, 'cancer_type': row.cancer_type, 'biological_context': row.biological_context, 'candidate_key': row.candidate_key, 'candidate_size': int(row.candidate_size), 'median_spot_cfar': float(row.median_spot_cfar), 'cfar_quantile': candidate_quantile[i], 'library': pathway.library, 'term': pathway.term, 'cfar_pathway_attribution': float(candidate_pathway[i, j]) if np.isfinite(candidate_pathway[i, j]) else np.nan, 'direct_pathway_local_contrast': float(direct_local[i, j]) if np.isfinite(direct_local[i, j]) else np.nan})
        graph = symmetric_knn_graph(data['coords'], k=6)
        (ring1_text, ring2_text) = ([], [])
        for row in meta.itertuples(index=False):
            core = [spot_lookup[x] for x in str(row.member_spots).split(';') if x in spot_lookup]
            (ring1, ring2) = graph_shells(graph, core)
            ring1_text.append(';'.join(data['spot_ids'][ring1].astype(str)))
            ring2_text.append(';'.join(data['spot_ids'][ring2].astype(str)))
        shell_matrices = {'CORE': core_matrix, 'RING1': mean_membership_matrix(pd.Series(ring1_text), spot_lookup, len(spot_lookup)), 'RING2': mean_membership_matrix(pd.Series(ring2_text), spot_lookup, len(spot_lookup)), 'LOCAL_BG': bg_matrix}
        for (shell, shell_matrix) in shell_matrices.items():
            direct_shell = np.asarray(shell_matrix @ spot_direct_pathway, dtype=np.float32)
            attr_shell = np.asarray(shell_matrix @ spot_attr_pathway, dtype=np.float32)
            nonempty = np.asarray(shell_matrix.sum(axis=1)).ravel() > 0
            for (j, pathway) in enumerate(selected.itertuples(index=False)):
                dv = direct_shell[nonempty, j]
                av = attr_shell[nonempty, j]
                gradient_rows.append({'sample': dataset_id, 'library': pathway.library, 'term': pathway.term, 'shell': shell, 'n_regions': int(nonempty.sum()), 'median_direct_pathway_score': float(np.nanmedian(dv)) if len(dv) else np.nan, 'mean_direct_pathway_score': float(np.nanmean(dv)) if len(dv) else np.nan, 'median_cfar_attributed_pathway_score': float(np.nanmedian(av)) if len(av) else np.nan, 'mean_cfar_attributed_pathway_score': float(np.nanmean(av)) if len(av) else np.nan})
        all_candidate_rows.append({'sample': dataset_id, 'n_candidates': len(meta), 'n_pathways': len(selected)})
        print(f'[candidate pathways] {dataset_id}: candidates={len(meta)} pathways={len(selected)}', flush=True)
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(PROJECT / 'section_cfar_pathway_correlations.csv', index=False)
    payload = pd.DataFrame(context_payload)
    payload.to_parquet(PROJECT / 'intermediate/candidate_pathway_attribution.parquet', index=False, compression='zstd')
    context = payload.groupby(['library', 'term', 'biological_context'], as_index=False).agg(n_regions=('candidate_key', 'nunique'), n_sections=('sample', 'nunique'), median_cfar_pathway_attribution=('cfar_pathway_attribution', 'median'), mean_cfar_pathway_attribution=('cfar_pathway_attribution', 'mean'), fraction_positive=('cfar_pathway_attribution', lambda x: float(np.mean(np.asarray(x) > 0))), median_direct_pathway_local_contrast=('direct_pathway_local_contrast', 'median'))
    context.to_csv(PROJECT / 'cfar_pathway_by_context.csv', index=False)
    cancer = payload.groupby(['library', 'term', 'cancer_type'], as_index=False).agg(n_regions=('candidate_key', 'nunique'), n_sections=('sample', 'nunique'), median_cfar_pathway_attribution=('cfar_pathway_attribution', 'median'), mean_cfar_pathway_attribution=('cfar_pathway_attribution', 'mean'), fraction_positive=('cfar_pathway_attribution', lambda x: float(np.mean(np.asarray(x) > 0))), median_direct_pathway_local_contrast=('direct_pathway_local_contrast', 'median'))
    cancer.to_csv(PROJECT / 'cfar_pathway_by_cancer.csv', index=False)
    gradient_summary = pd.DataFrame(gradient_rows)
    gradient_summary.to_csv(PROJECT / 'cfar_spatial_gradient.csv', index=False)
    return (correlations, payload, context, cancer)

def summarize_pathway_associations(correlations: pd.DataFrame, payload: pd.DataFrame, selected: pd.DataFrame) -> None:
    rows = []
    for ((library, term), group) in correlations.groupby(['library', 'term'], sort=False):
        values = group['spearman_rho'].to_numpy(float)
        values = values[np.isfinite(values)]
        (low, high) = section_bootstrap_ci(values, f'{library}|{term}')
        rows.append({'library': library, 'term': term, 'n_sections': len(values), 'median_rho': float(np.median(values)) if len(values) else np.nan, 'mean_rho': float(np.mean(values)) if len(values) else np.nan, 'fraction_rho_positive': float(np.mean(values > 0)) if len(values) else np.nan, 'section_bootstrap_ci_low': low, 'section_bootstrap_ci_high': high, 'section_bootstrap_replicates': N_BOOT})
    summary = pd.DataFrame(rows)
    summary.to_csv(PROJECT / 'intermediate/cfar_pathway_correlation_summary.csv', index=False)
    mixed_rows = []
    mixed_terms = selected.sort_values(['block_fdr', 'region_fdr', 'term'], na_position='last', kind='mergesort').head(15)
    for row in mixed_terms.itertuples(index=False):
        frame = payload[payload['library'].eq(row.library) & payload['term'].eq(row.term)].dropna(subset=['direct_pathway_local_contrast', 'median_spot_cfar']).copy()
        (status, beta, se, p, n) = ('not_fit', np.nan, np.nan, np.nan, len(frame))
        if smf is not None and len(frame) >= 50 and (frame['sample'].nunique() >= 3):
            try:
                model = smf.mixedlm('direct_pathway_local_contrast ~ median_spot_cfar', frame, groups=frame['sample'])
                fit = model.fit(reml=True, method='lbfgs', maxiter=300, disp=False)
                beta = float(fit.params.get('median_spot_cfar', np.nan))
                se = float(fit.bse.get('median_spot_cfar', np.nan))
                p = float(fit.pvalues.get('median_spot_cfar', np.nan))
                status = 'mixed_random_intercept'
            except Exception as exc:
                status = f'fit_failed:{type(exc).__name__}'
        mixed_rows.append({'library': row.library, 'term': row.term, 'n_candidates': n, 'n_sections': frame['sample'].nunique(), 'beta_cfar_strength': beta, 'se': se, 'p_value': p, 'model_status': status, 'model': 'direct pathway local contrast ~ median member-spot CFAR + (1|section)', 'role': 'secondary sensitivity'})
    mixed = pd.DataFrame(mixed_rows)
    mixed['fdr'] = bh_fdr(mixed['p_value'])
    mixed.to_csv(PROJECT / 'cfar_pathway_mixed_model.csv', index=False)
    quantile = payload.groupby(['library', 'term', 'sample', 'cfar_quantile'], observed=True, as_index=False).agg(n_candidates=('candidate_key', 'nunique'), median_cfar_pathway_attribution=('cfar_pathway_attribution', 'median'), median_direct_pathway_local_contrast=('direct_pathway_local_contrast', 'median'), mean_direct_pathway_local_contrast=('direct_pathway_local_contrast', 'mean'))
    quantile.to_csv(PROJECT / 'cfar_quantile_pathway_summary.csv', index=False)

def finalize_pathway_classes(master: pd.DataFrame, context: pd.DataFrame, cancer: pd.DataFrame) -> pd.DataFrame:
    result = master.copy()
    result['class_support_detail'] = ''
    for (i, row) in result.iterrows():
        if str(row['pathway_class']).startswith(('A ', 'B ')):
            continue
        ctx = context[context['library'].eq(row['library']) & context['term'].eq(row['term'])].copy()
        ctx['direction_fraction'] = np.maximum(ctx['fraction_positive'], 1 - ctx['fraction_positive'])
        ctx_hit = ctx[(ctx['n_regions'] >= 8) & (ctx['median_cfar_pathway_attribution'].abs() >= 0.05) & (ctx['direction_fraction'] >= 0.75)]
        if not ctx_hit.empty:
            best = ctx_hit.sort_values(['median_cfar_pathway_attribution', 'biological_context'], key=lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x, ascending=False).iloc[0]
            result.at[i, 'pathway_class'] = 'C CONTEXT-ASSOCIATED CFAR PROGRAM'
            result.at[i, 'class_support_detail'] = f"{best['biological_context']}; n={int(best['n_regions'])}"
            continue
        can = cancer[cancer['library'].eq(row['library']) & cancer['term'].eq(row['term'])].copy()
        can['direction_fraction'] = np.maximum(can['fraction_positive'], 1 - can['fraction_positive'])
        can_hit = can[(can['n_sections'] >= 2) & (can['median_cfar_pathway_attribution'].abs() >= 0.05) & (can['direction_fraction'] >= 0.75)]
        if not can_hit.empty:
            best = can_hit.sort_values('median_cfar_pathway_attribution', key=lambda x: x.abs(), ascending=False).iloc[0]
            result.at[i, 'pathway_class'] = 'D CANCER-ASSOCIATED CFAR PROGRAM'
            result.at[i, 'class_support_detail'] = f"{best['cancer_type']}; sections={int(best['n_sections'])}"
    result.to_csv(PROJECT / 'intermediate/cfar_pathway_classification_master.csv', index=False)
    for library in ['hallmark', 'reactome', 'gobp']:
        classes = result[result['library'].eq(library)][['term', 'pathway_class', 'class_support_detail']]
        for suffix in ['regionweighted', 'sectionequal', 'blockperm']:
            path = PROJECT / f'{library}_cfar_{suffix}.csv'
            frame = pd.read_csv(path)
            frame = frame.drop(columns=['pathway_class', 'class_support_detail'], errors='ignore').merge(classes, on='term', how='left')
            frame.to_csv(path, index=False)
    return result

def random_connected_set(graph: dict[int, set[int]], size: int, rng: np.random.Generator) -> list[int] | None:
    vertices = np.asarray(sorted(graph), dtype=int)
    for _ in range(100):
        seed = int(rng.choice(vertices))
        selected = {seed}
        frontier = set(graph[seed])
        while len(selected) < size and frontier:
            choices = np.asarray(sorted(frontier), dtype=int)
            nxt = int(rng.choice(choices))
            frontier.discard(nxt)
            if nxt in selected:
                continue
            selected.add(nxt)
            frontier.update(graph[nxt] - selected)
        if len(selected) == size:
            return sorted(selected)
    return None

def random_calibration(candidate_meta: pd.DataFrame, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]], top_n_pathways: int=15) -> pd.DataFrame:
    selected_top = selected.sort_values(['block_fdr', 'region_fdr', 'term'], na_position='last', kind='mergesort').head(top_n_pathways).reset_index(drop=True)
    rows = []
    for (dataset_id, meta) in candidate_meta.groupby('sample', sort=False):
        with np.load(PROJECT / 'intermediate/samples' / f'{dataset_id}.npz', allow_pickle=False) as archive:
            data = {key: archive[key] for key in ('coords', 'genes', 'cfar', 'gene_signed')}
        cand = np.load(PROJECT / 'intermediate/candidates' / f'{dataset_id}.npz', allow_pickle=True)
        graph = symmetric_knn_graph(data['coords'], k=6)
        (full_membership, full_sizes) = pathway_membership(data['genes'].astype(str), selected, libraries)
        top_positions = selected.set_index(['library', 'term'])['pathway_index'].reindex(pd.MultiIndex.from_frame(selected_top[['library', 'term']])).to_numpy(int)
        candidate_pathway = set_contrast_from_ranks(centered_row_ranks(cand['signed']), full_membership, full_sizes)[:, top_positions]
        for size in sorted(meta['candidate_size'].unique()):
            subset = meta['candidate_size'].eq(size).to_numpy()
            rng = np.random.default_rng(stable_seed(dataset_id, int(size), 'matched_random', base=RANDOM_BASE_SEED))
            random_sets = []
            attempts = 0
            while len(random_sets) < N_RANDOM and attempts < N_RANDOM * 20:
                attempts += 1
                value = random_connected_set(graph, int(size), rng)
                if value is not None:
                    random_sets.append(value)
            if len(random_sets) < N_RANDOM:
                raise RuntimeError(f'Random reconstruction did not reach {N_RANDOM}: {dataset_id}, size {size}')
            random_cfar = np.asarray([np.median(data['cfar'][idx]) for idx in random_sets], dtype=float)
            random_gene = np.asarray([np.median(data['gene_signed'][idx], axis=0) for idx in random_sets], dtype=np.float32)
            random_pathway = set_contrast_from_ranks(centered_row_ranks(random_gene), full_membership, full_sizes)[:, top_positions]
            observed_cfar = meta.loc[subset, 'median_spot_cfar'].to_numpy(float)
            observed_median = float(np.median(observed_cfar))
            rows.append({'sample': dataset_id, 'candidate_size': int(size), 'metric': 'CFAR intensity', 'library': '', 'term': '', 'n_observed_candidates': int(subset.sum()), 'n_matched_random_regions': len(random_sets), 'observed_median': observed_median, 'random_median': float(np.median(random_cfar)), 'observed_minus_random_median': observed_median - float(np.median(random_cfar)), 'candidate_percentile_in_random_distribution': 100 * float(np.mean(random_cfar <= observed_median)), 'empirical_random_p_two_sided': (1 + int(np.sum(np.abs(random_cfar - np.median(random_cfar)) >= abs(observed_median - np.median(random_cfar))))) / (1 + len(random_cfar)), 'random_framework': 'deterministic pooled-landscape reconstruction: symmetric 6-NN, 500 connected regions, base seed 20260805', 'historical_member_identity_verifiable': False})
            for (j, pathway) in enumerate(selected_top.itertuples(index=False)):
                observed = np.abs(candidate_pathway[subset, j])
                null = np.abs(random_pathway[:, j])
                observed_median = float(np.nanmedian(observed))
                random_median = float(np.nanmedian(null))
                rows.append({'sample': dataset_id, 'candidate_size': int(size), 'metric': 'absolute CFAR-attributed pathway score', 'library': pathway.library, 'term': pathway.term, 'n_observed_candidates': int(np.isfinite(observed).sum()), 'n_matched_random_regions': int(np.isfinite(null).sum()), 'observed_median': observed_median, 'random_median': random_median, 'observed_minus_random_median': observed_median - random_median, 'candidate_percentile_in_random_distribution': 100 * float(np.nanmean(null <= observed_median)), 'empirical_random_p_two_sided': (1 + int(np.nansum(np.abs(null - random_median) >= abs(observed_median - random_median)))) / (1 + int(np.isfinite(null).sum())), 'random_framework': 'deterministic pooled-landscape reconstruction: symmetric 6-NN, 500 connected regions, base seed 20260805', 'historical_member_identity_verifiable': False})
        print(f"[random] {dataset_id}: sizes={sorted(meta['candidate_size'].unique())}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / 'cfar_random_calibration.csv', index=False)
    return result
