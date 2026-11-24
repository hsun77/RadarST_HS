from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict,deque
import math,hashlib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor


SEED = 20260811
N_SEARCH_ATTEMPTS = 96

def token_channel(stage: str, proposal_type: str) -> str:
    if stage == 'additive_signed_utility_tail':
        return 'tail'
    if proposal_type == 'singleton':
        return 'singleton'
    return 'core'

def exact_token_assignment(frame: pd.DataFrame, teacher_scores: np.ndarray, utility_scores: np.ndarray, formal: pd.DataFrame, spot_ids: list[str], method: str, variant: str) -> tuple[dict[str, str], pd.DataFrame, dict[str, object]]:
    records = R.candidate_records(frame, spot_ids)
    (core_mask, singleton_mask, tail_mask, route) = R.eligibility_masks(frame)
    masks = {'core': core_mask, 'singleton': singleton_mask, 'tail': tail_mask}
    score_arrays = {'core': np.asarray(teacher_scores, dtype=float), 'singleton': np.asarray(teacher_scores, dtype=float), 'tail': np.asarray(utility_scores, dtype=float)}
    accepted_columns = ['dataset_id', 'sample_id', 'method', 'variant', 'candidate_id', 'candidate_key', 'proposal_type', 'member_spots', 'n_spots', 'channel', 'selection_score', 'topology_valid', 'fragmentation_risk', 'erosion_risk', 'boundary_leakage_risk']
    if formal.empty:
        return ({}, pd.DataFrame(columns=accepted_columns), {'safety_route': route, 'search_attempt_selected': 0, 'formal_modified_spots': 0, 'ablation_modified_spots': 0, 'exact_token_histogram_match': True})
    pools: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        idx = int(record['index'])
        for channel in ('core', 'singleton', 'tail'):
            if masks[channel][idx]:
                item = dict(record)
                item['score'] = float(score_arrays[channel][idx])
                pools[channel, str(record['proposal_type']), int(record['n_spots'])].append(item)
    for options in pools.values():
        options.sort(key=lambda item: (-float(item['score']), str(item['candidate_key'])))
    tokens = [(token_channel(str(row.stage), str(row.proposal_type)), str(row.proposal_type), int(row.n_spots)) for row in formal.itertuples(index=False)]
    tokens.sort(key=lambda item: (-item[2], item[0], item[1]))
    expected_hist = pd.Series(tokens).value_counts().sort_index().to_dict()
    base_seed = SEED + sum((ord(char) for char in f'{variant}|{method}|{formal.dataset_id.iloc[0]}'))
    rng = np.random.default_rng(base_seed)
    best = None
    for attempt in range(N_SEARCH_ATTEMPTS):
        touched: set[str] = set()
        selected = []
        total_score = 0.0
        success = True
        for token in tokens:
            options = pools.get(token, [])
            width_limit = 1 if attempt == 0 else 32
            available = []
            for item in options:
                if item['members'] & touched:
                    continue
                available.append(item)
                if len(available) >= width_limit:
                    break
            if not available:
                success = False
                break
            if attempt == 0:
                choice = available[0]
            else:
                width = len(available)
                ranks = np.arange(width, dtype=float)
                probability = np.exp(-ranks / 5.0)
                probability /= probability.sum()
                choice = available[int(rng.choice(width, p=probability))]
            selected.append((token[0], choice))
            touched.update(choice['members'])
            total_score += float(choice['score'])
        if success and (best is None or total_score > best[0]):
            best = (total_score, selected, attempt)
    if best is None:
        raise RuntimeError(f'{variant} {formal.dataset_id.iloc[0]} {method}: exact-token search failed')
    (_score, selected, successful_attempt) = best
    assignment: dict[str, str] = {}
    accepted_rows = []
    observed_tokens = []
    for (number, (channel, item)) in enumerate(selected, start=1):
        label = f'RadarST_ablation__{variant}__{method}__{number}'
        assignment.update({spot: label for spot in item['members']})
        observed_tokens.append((channel, item['proposal_type'], item['n_spots']))
        row = frame.iloc[int(item['index'])]
        accepted_rows.append({'dataset_id': str(row.dataset_id), 'sample_id': str(row.sample_id), 'method': method, 'variant': variant, 'candidate_id': item['candidate_id'], 'candidate_key': item['candidate_key'], 'proposal_type': item['proposal_type'], 'member_spots': ';'.join(sorted(item['members'])), 'n_spots': item['n_spots'], 'channel': channel, 'selection_score': item['score'], 'topology_valid': bool(row.topology_valid), 'fragmentation_risk': float(row.fragmentation_risk), 'erosion_risk': float(row.erosion_risk), 'boundary_leakage_risk': float(row.boundary_leakage_risk)})
    observed_hist = pd.Series(observed_tokens).value_counts().sort_index().to_dict()
    expected_k = int(formal.n_spots.sum())
    if observed_hist != expected_hist or len(assignment) != expected_k:
        raise RuntimeError(f'{variant} {formal.dataset_id.iloc[0]} {method}: exact budget/composition mismatch')
    return (assignment, pd.DataFrame(accepted_rows, columns=accepted_columns), {'safety_route': route, 'search_attempt_selected': int(successful_attempt), 'formal_modified_spots': expected_k, 'ablation_modified_spots': int(len(assignment)), 'exact_token_histogram_match': True})
