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
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,normalized_mutual_info_score


EPS = 1e-12
BAD_TRUTH = {'', 'nan', 'na', 'none', 'null', 'unknown', '<na>', 'unassigned'}

def parse_spots(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    return [x for x in text.split(';') if x] if text else []

def strict_metrics(truth: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    truth = truth.astype(str)
    pred = pred.astype(str)
    mask = ~truth.str.lower().isin(BAD_TRUTH)
    t = truth[mask]
    p = pred[mask]
    if len(t) == 0 or t.nunique() <= 1 or p.nunique() <= 1:
        return {'ARI': np.nan, 'NMI': np.nan, 'AMI': np.nan, 'n_pred_clusters': int(p.nunique())}
    return {'ARI': float(adjusted_rand_score(t, p)), 'NMI': float(normalized_mutual_info_score(t, p)), 'AMI': float(adjusted_mutual_info_score(t, p)), 'n_pred_clusters': int(p.nunique())}

def label_connected_components(labels: pd.Series, spot_ids: list[str], adjacency: list[list[int]]) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    label_values = labels.reindex(spot_ids).fillna('missing_pred').astype(str).to_numpy()
    by_label: dict[str, list[int]] = defaultdict(list)
    for (idx, label) in enumerate(label_values):
        by_label[str(label)].append(idx)
    component_sets: dict[str, set[str]] = {}
    spot_to_component: dict[str, str] = {}
    label_to_components: dict[str, list[str]] = defaultdict(list)
    for (label, indices) in by_label.items():
        allowed = set(indices)
        seen: set[int] = set()
        component_number = 0
        for start in indices:
            if start in seen:
                continue
            component_number += 1
            queue = deque([start])
            seen.add(start)
            members: list[int] = []
            while queue:
                node = queue.popleft()
                members.append(node)
                for neighbor in adjacency[node]:
                    if neighbor in allowed and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            component_id = f'{label}::cc{component_number:04d}'
            member_spots = {spot_ids[i] for i in members}
            component_sets[component_id] = member_spots
            label_to_components[label].append(component_id)
            for spot in member_spots:
                spot_to_component[spot] = component_id
    return (component_sets, spot_to_component, dict(label_to_components))

def true_best_match_from_regions(regions: dict[str, set[str]], truth_set: set[str]) -> dict[str, object]:
    candidates = []
    for (region_id, region) in regions.items():
        overlap = len(region & truth_set)
        if overlap == 0:
            continue
        iou = overlap / max(1, len(region | truth_set))
        candidates.append((float(iou), overlap, -len(region), str(region_id), len(region)))
    if not candidates:
        return {'match_id': '', 'overlap': 0, 'region_size': 0, 'iou': 0.0}
    (iou, overlap, _neg_size, region_id, region_size) = max(candidates)
    return {'match_id': region_id, 'overlap': int(overlap), 'region_size': int(region_size), 'iou': float(iou)}
