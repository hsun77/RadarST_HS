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

def percentile_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    values = values.fillna(float(values.median()) if values.notna().any() else 0.0)
    return values.rank(method='average', pct=True)

def canonical_spots(value: str) -> str:
    return ';'.join(sorted(set(B67.parse_spots(str(value)))))

def candidate_key(dataset: str, proposal_type: str, member_spots: str) -> str:
    return f'{dataset}||{proposal_type}||{canonical_spots(member_spots)}'

def add_multiscale_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    family_count = max(1, result['proposal_type'].nunique())
    spot_families: dict[str, set[str]] = defaultdict(set)
    for row in result.itertuples(index=False):
        for spot in B67.parse_spots(row.canonical_member_spots):
            spot_families[str(spot)].add(str(row.proposal_type))
    mean_support = []
    minimum_support = []
    for members in result['canonical_member_spots']:
        supports = [len(spot_families[spot]) / family_count for spot in B67.parse_spots(members)]
        mean_support.append(float(np.mean(supports)) if supports else 0.0)
        minimum_support.append(float(np.min(supports)) if supports else 0.0)
    result['multiscale_persistence_mean'] = mean_support
    result['multiscale_persistence_min'] = minimum_support
    return result

def prediction_codes(base: pd.Series, spot_ids: list[str]) -> np.ndarray:
    values = base.reindex(spot_ids).fillna('missing_pred').astype(str)
    (codes, _) = pd.factorize(values, sort=False)
    return codes.astype(np.int32)[:, None]

def label_features(indices: set[int], labels: np.ndarray, cluster_sizes: dict[int, int]) -> dict[str, float]:
    if not indices:
        return {'candidate_label_purity': 0.0, 'candidate_label_entropy': 1.0, 'dominant_cluster_to_candidate_ratio': 1000000.0}
    (values, counts) = np.unique(labels[np.fromiter(sorted(indices), dtype=int)], return_counts=True)
    dominant_position = int(np.argmax(counts))
    dominant = int(values[dominant_position])
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities + EPS)).sum())
    normalized_entropy = entropy / max(EPS, math.log(max(2, len(counts))))
    return {'candidate_label_purity': float(counts[dominant_position] / counts.sum()), 'candidate_label_entropy': float(normalized_entropy), 'dominant_cluster_to_candidate_ratio': float(cluster_sizes[dominant] / max(1, len(indices)))}
