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


EPS = 1e-8

def edge_distances(pcs: np.ndarray, adjacency: list[list[int]]) -> tuple[dict[tuple[int, int], float], np.ndarray]:
    distances: dict[tuple[int, int], float] = {}
    anomaly = np.zeros(len(adjacency), dtype=np.float32)
    scale = math.sqrt(max(1, pcs.shape[1]))
    for i, neighbors in enumerate(adjacency):
        values = []
        for j in neighbors:
            key = (i, j) if i < j else (j, i)
            if key not in distances:
                distances[key] = float(np.linalg.norm(pcs[i] - pcs[j]) / scale)
            values.append(distances[key])
        anomaly[i] = float(np.mean(values)) if values else 0.0
    ranks = pd.Series(anomaly).rank(method="average", pct=True).to_numpy(np.float32)
    return distances, ranks

def q75(values: list[float]) -> float:
    return float(np.quantile(values, 0.75)) if values else 0.0

def candidate_features(
    members: set[int],
    pcs: np.ndarray,
    adjacency: list[list[int]],
    edge_distance: dict[tuple[int, int], float],
    local_anomaly_rank: np.ndarray,
) -> dict[str, float]:
    if not members:
        return {}
    background: set[int] = set()
    internal_edges: set[tuple[int, int]] = set()
    boundary_edges: set[tuple[int, int]] = set()
    degrees = []
    for i in members:
        degrees.append(len(adjacency[i]))
        for j in adjacency[i]:
            key = (i, j) if i < j else (j, i)
            if j in members:
                internal_edges.add(key)
            else:
                boundary_edges.add(key)
                background.add(j)
    if not background:
        for i in members:
            for j in adjacency[i]:
                for k in adjacency[j]:
                    if k not in members:
                        background.add(k)
    member_index = np.fromiter(sorted(members), dtype=int)
    background_index = np.fromiter(sorted(background), dtype=int)
    target = pcs[member_index]
    target_mean = target.mean(axis=0)
    if len(background_index):
        bg = pcs[background_index]
        bg_mean = bg.mean(axis=0)
        target_background_distance = float(
            np.linalg.norm(target_mean - bg_mean) / math.sqrt(max(1, pcs.shape[1]))
        )
        target_norm = float(np.linalg.norm(target_mean))
        bg_norm = float(np.linalg.norm(bg_mean))
        cosine = 1.0 - float(np.dot(target_mean, bg_mean) / max(EPS, target_norm * bg_norm))
        bg_dispersion = float(np.mean(np.linalg.norm(bg - bg_mean, axis=1)))
    else:
        target_background_distance = 0.0
        cosine = 0.0
        bg_dispersion = 0.0
    target_dispersion = float(np.mean(np.linalg.norm(target - target_mean, axis=1)))
    boundary_values = [edge_distance[key] for key in boundary_edges]
    internal_values = [edge_distance[key] for key in internal_edges]
    boundary_mean = float(np.mean(boundary_values)) if boundary_values else 0.0
    internal_mean = float(np.mean(internal_values)) if internal_values else 0.0
    anomalies = local_anomaly_rank[member_index]
    possible_internal = max(1.0, len(members) * (len(members) - 1) / 2)
    cut = len(boundary_edges)
    internal_twice = 2 * len(internal_edges)
    return {
        "expr_target_background_distance": target_background_distance,
        "expr_target_background_cosine": cosine,
        "expr_target_internal_dispersion": target_dispersion,
        "expr_background_dispersion": bg_dispersion,
        "expr_boundary_edge_mean": boundary_mean,
        "expr_boundary_edge_q75": q75(boundary_values),
        "expr_boundary_edge_max": float(max(boundary_values)) if boundary_values else 0.0,
        "expr_internal_edge_mean": internal_mean,
        "expr_boundary_internal_ratio": boundary_mean / max(EPS, internal_mean),
        "expr_local_anomaly_mean": float(np.mean(anomalies)),
        "expr_local_anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "expr_local_anomaly_max": float(np.max(anomalies)),
        "graph_internal_edge_density": len(internal_edges) / possible_internal,
        "graph_boundary_edges_per_spot": cut / max(1, len(members)),
        "graph_conductance": cut / max(1, cut + internal_twice),
        "graph_background_spots": float(len(background)),
        "graph_candidate_degree_mean": float(np.mean(degrees)) if degrees else 0.0,
    }
