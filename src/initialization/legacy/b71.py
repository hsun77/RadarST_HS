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


EPS = 1e-12

def apply_relaxed_evidence_gate(candidates: pd.DataFrame, protocol: dict[str, object]) -> pd.DataFrame:
    result = candidates.copy()
    gates = protocol['relaxed_truth_independent_evidence_gate']
    singleton = gates['singleton']
    multispot = gates['multispot']
    singleton_pass = result['is_singleton'] & result['boundary_percentile'].ge(float(singleton['boundary_percentile_min'])) & result['proposal_percentile'].ge(float(singleton['proposal_percentile_min'])) & (result['island_support_methods'].ge(int(singleton['minimum_island_support_methods'])) | result['rare_support_methods'].ge(int(singleton['minimum_rare_support_methods_alternative'])))
    multispot_pass = ~result['is_singleton'] & result['boundary_percentile'].ge(float(multispot['boundary_percentile_min'])) & result['proposal_percentile'].ge(float(multispot['proposal_percentile_min'])) & (result['island_support_methods'].ge(int(multispot['minimum_island_support_methods'])) | result['rare_support_methods'].ge(int(multispot['minimum_rare_support_methods_alternative'])))
    result['passes_71A_evidence_gate'] = singleton_pass | multispot_pass
    return result

def is_connected(indices: set[int], adjacency: list[list[int]]) -> bool:
    if not indices:
        return False
    start = next(iter(indices))
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in indices and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(indices)

def prepare_dataset_candidates(candidates: pd.DataFrame, context: dict[str, object]) -> tuple[list[dict[str, object]], Counter[str]]:
    spot_ids = list(map(str, context['spot_ids']))
    spot_to_index = {spot: index for (index, spot) in enumerate(spot_ids)}
    adjacency = context['adjacency']
    rows: list[dict[str, object]] = []
    audit: Counter[str] = Counter()
    for row in candidates[candidates['passes_71A_evidence_gate']].itertuples(index=False):
        spots = list(dict.fromkeys(B67.parse_spots(row.member_spots)))
        if not spots or any((spot not in spot_to_index for spot in spots)):
            audit['invalid_or_empty_spots'] += 1
            continue
        indices = {spot_to_index[spot] for spot in spots}
        if not is_connected(indices, adjacency):
            audit['candidate_not_connected'] += 1
            continue
        rows.append({'candidate_id': str(row.candidate_id), 'proposal_type': str(row.proposal_type), 'member_spots': ';'.join(spots), 'indices': indices, 'n_spots': len(indices), 'evidence_score': float(row.evidence_score), 'boundary_percentile': float(row.boundary_percentile), 'proposal_percentile': float(row.proposal_percentile), 'island_support_methods': int(row.island_support_methods), 'rare_support_methods': int(row.rare_support_methods)})
    audit['raw_candidates'] = len(candidates)
    audit['evidence_gate_failed'] = int((~candidates['passes_71A_evidence_gate']).sum())
    audit['evidence_gate_passed'] = int(candidates['passes_71A_evidence_gate'].sum())
    audit['connected_candidates'] = len(rows)
    return (rows, audit)

def residual_component_count(component: set[int], removed: set[int], adjacency: list[list[int]]) -> int:
    residual_size = len(component) - len(removed)
    if residual_size <= 0:
        return 0
    boundary = {neighbor for node in removed for neighbor in adjacency[node] if neighbor in component and neighbor not in removed}
    if not boundary:
        return 1
    unseen_boundary = set(boundary)
    n_components = 0
    while unseen_boundary:
        n_components += 1
        start = next(iter(unseen_boundary))
        seen = {start}
        queue = deque([start])
        unseen_boundary.discard(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in removed or neighbor not in component or neighbor in seen:
                    continue
                seen.add(neighbor)
                unseen_boundary.discard(neighbor)
                queue.append(neighbor)
            if not unseen_boundary:
                return n_components
    return n_components

def method_component_index(base: pd.Series, context: dict[str, object]) -> tuple[dict[str, set[int]], list[str], np.ndarray]:
    spot_ids = list(map(str, context['spot_ids']))
    spot_to_index = {spot: index for (index, spot) in enumerate(spot_ids)}
    (component_sets, spot_to_component, _) = B67.label_connected_components(base, spot_ids, context['adjacency'])
    components = {component_id: {spot_to_index[spot] for spot in spots} for (component_id, spots) in component_sets.items()}
    component_id_by_index = [spot_to_component[spot] for spot in spot_ids]
    labels = base.reindex(spot_ids).fillna('missing_pred').astype(str).to_numpy()
    return (components, component_id_by_index, labels)

def score_method_candidates(prepared: list[dict[str, object]], base: pd.Series, context: dict[str, object], protocol: dict[str, object]) -> tuple[list[dict[str, object]], Counter[str]]:
    adjacency = context['adjacency']
    (components, component_id_by_index, labels) = method_component_index(base, context)
    minimum_utility = float(protocol['utility']['minimum_utility'])
    scored: list[dict[str, object]] = []
    audit: Counter[str] = Counter()
    for candidate in prepared:
        indices = candidate['indices']
        counts = Counter((component_id_by_index[index] for index in indices))
        if any((count == len(components[component_id]) for (component_id, count) in counts.items())):
            audit['complete_upstream_component_relabel'] += 1
            continue
        dominant_fraction = max(counts.values()) / len(indices)
        cross_component_risk = 1.0 - dominant_fraction
        erosion_risk = max((count / len(components[component_id]) for (component_id, count) in counts.items()))
        outside_labels = [str(labels[neighbor]) for index in indices for neighbor in adjacency[index] if neighbor not in indices]
        if outside_labels:
            dominant_boundary = Counter(outside_labels).most_common(1)[0][1] / len(outside_labels)
            boundary_leakage_risk = 1.0 - dominant_boundary
        else:
            dominant_boundary = 0.0
            boundary_leakage_risk = 1.0
        optimistic_safety = 1.0 - np.mean([0.0, cross_component_risk, boundary_leakage_risk, erosion_risk])
        optimistic_utility = 0.55 * candidate['evidence_score'] + 0.45 * optimistic_safety
        if optimistic_utility < minimum_utility - EPS:
            audit['utility_upper_bound_below_minimum'] += 1
            continue
        residual_extra_components = 0
        for component_id in counts:
            component = components[component_id]
            removed = indices & component
            residual_extra_components += max(0, residual_component_count(component, removed, adjacency) - 1)
        fragmentation_risk = min(1.0, residual_extra_components / 2.0)
        safety_score = float(np.clip(1.0 - np.mean([fragmentation_risk, cross_component_risk, boundary_leakage_risk, erosion_risk]), 0.0, 1.0))
        utility = 0.55 * candidate['evidence_score'] + 0.45 * safety_score
        if utility < minimum_utility - EPS:
            audit['utility_below_minimum'] += 1
            continue
        scored.append({**candidate, 'fragmentation_risk': fragmentation_risk, 'residual_extra_components': residual_extra_components, 'cross_component_risk': cross_component_risk, 'boundary_leakage_risk': boundary_leakage_risk, 'dominant_boundary_label_fraction': dominant_boundary, 'erosion_risk': erosion_risk, 'safety_score': safety_score, 'writeback_utility': utility, 'touched_upstream_components': len(counts)})
    scored.sort(key=lambda row: (-row['writeback_utility'], -row['evidence_score'], -row['safety_score'], row['candidate_id']))
    audit['risk_scored_and_utility_passed'] = len(scored)
    return (scored, audit)

def accept_scored_candidates(scored: list[dict[str, object]], base: pd.Series, context: dict[str, object], method: str, cap: float) -> tuple[pd.DataFrame, pd.Series, Counter[str]]:
    spot_ids = list(map(str, context['spot_ids']))
    labels = base.reindex(spot_ids).fillna('missing_pred').astype(str).to_numpy(copy=True)
    maximum_modified = int(math.floor(len(spot_ids) * cap))
    touched: set[int] = set()
    accepted_rows: list[dict[str, object]] = []
    audit: Counter[str] = Counter()
    for candidate in scored:
        indices = candidate['indices']
        if indices & touched:
            audit['overlap_with_higher_utility_candidate'] += 1
            continue
        if len(touched) + len(indices) > maximum_modified:
            audit['safety_cap_atomic_rejection'] += 1
            continue
        candidate_label = f"RadarST_RCW__{method}__{candidate['candidate_id']}"
        for index in indices:
            labels[index] = candidate_label
        touched.update(indices)
        accepted_rows.append({'dataset_id': context['dataset_id'], 'sample_id': context['sample_id'], 'method': method, 'cap': cap, 'candidate_id': candidate['candidate_id'], 'candidate_label': candidate_label, 'proposal_type': candidate['proposal_type'], 'n_spots': candidate['n_spots'], 'member_spots': candidate['member_spots'], 'evidence_score': candidate['evidence_score'], 'safety_score': candidate['safety_score'], 'writeback_utility': candidate['writeback_utility'], 'fragmentation_risk': candidate['fragmentation_risk'], 'residual_extra_components': candidate['residual_extra_components'], 'cross_component_risk': candidate['cross_component_risk'], 'boundary_leakage_risk': candidate['boundary_leakage_risk'], 'dominant_boundary_label_fraction': candidate['dominant_boundary_label_fraction'], 'erosion_risk': candidate['erosion_risk'], 'touched_upstream_components': candidate['touched_upstream_components'], 'boundary_percentile': candidate['boundary_percentile'], 'proposal_percentile': candidate['proposal_percentile'], 'island_support_methods': candidate['island_support_methods'], 'rare_support_methods': candidate['rare_support_methods'], 'modified_spots_after_acceptance': len(touched)})
        if len(touched) >= maximum_modified:
            break
    audit['accepted'] = len(accepted_rows)
    audit['modified_spots'] = len(touched)
    accepted = pd.DataFrame(accepted_rows)
    augmented = pd.Series(labels, index=spot_ids, dtype=str)
    return (accepted, augmented, audit)
