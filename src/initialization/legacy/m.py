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


EPS = 1e-09

def map_summary(labels: np.ndarray, adjacency: list[list[int]]) -> dict[str, float]:
    counts = Counter(labels.tolist())
    fractions = np.asarray(list(counts.values()), dtype=float) / max(1, len(labels))
    entropy = float(-(fractions * np.log(fractions + EPS)).sum() / max(EPS, math.log(max(2, len(fractions)))))
    same_edges = 0
    all_edges = 0
    boundary_spots = 0
    for (i, neighbors) in enumerate(adjacency):
        different = False
        for j in neighbors:
            if i < j:
                all_edges += 1
                same_edges += int(labels[i] == labels[j])
            different |= labels[i] != labels[j]
        boundary_spots += int(different)
    return {'map_log_n_clusters': float(np.log1p(len(counts))), 'map_cluster_entropy': entropy, 'map_largest_cluster_fraction': float(fractions.max()), 'map_median_cluster_fraction': float(np.median(fractions)), 'map_same_label_edge_fraction': same_edges / max(1, all_edges), 'map_boundary_spot_fraction': boundary_spots / max(1, len(labels))}

def candidate_context(members: set[int], labels: np.ndarray, adjacency: list[list[int]], cluster_sizes: dict[str, int], cluster_percentiles: dict[str, float], summary: dict[str, float]) -> dict[str, float]:
    if not members:
        return dict(summary)
    member_labels = [str(labels[i]) for i in members]
    (dominant, dominant_count) = Counter(member_labels).most_common(1)[0]
    internal_edges = 0
    internal_same = 0
    boundary_edges = 0
    boundary_cross = 0
    member_boundary = 0
    for i in members:
        has_different = False
        for j in adjacency[i]:
            if j in members:
                if i < j:
                    internal_edges += 1
                    internal_same += int(labels[i] == labels[j])
            else:
                boundary_edges += 1
                boundary_cross += int(str(labels[j]) != dominant)
            has_different |= labels[i] != labels[j]
        member_boundary += int(has_different)
    result = dict(summary)
    result.update({'upstream_dominant_cluster_fraction': cluster_sizes[dominant] / max(1, len(labels)), 'upstream_dominant_cluster_size_percentile': cluster_percentiles[dominant], 'upstream_candidate_fraction_of_cluster': len(members) / max(1, cluster_sizes[dominant]), 'upstream_candidate_dominant_purity': dominant_count / max(1, len(members)), 'upstream_internal_same_label_fraction': internal_same / max(1, internal_edges), 'upstream_boundary_cross_label_fraction': boundary_cross / max(1, boundary_edges), 'upstream_member_boundary_fraction': member_boundary / max(1, len(members)), 'upstream_boundary_edges_per_spot': boundary_edges / max(1, len(members))})
    return result
