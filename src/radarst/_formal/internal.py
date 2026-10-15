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


N_PCS = 20

def spots_to_text(spots: list[str]) -> str:
    return ";".join(map(str, spots))

def median_nn_scale(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 1.0
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist[dist == 0] = np.inf
    vals = np.min(dist, axis=1)
    med = float(np.nanmedian(vals))
    return med if np.isfinite(med) and med > 0 else 1.0

def adjacency(coords: np.ndarray) -> list[set[int]]:
    n = len(coords)
    adj = [set() for _ in range(n)]
    if n <= 1:
        return adj
    scale = median_nn_scale(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    threshold = scale * 1.55
    for i in range(n):
        near = np.where((dist[i] > 0) & (dist[i] <= threshold))[0]
        for j in near:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj

def pca_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = x / np.maximum(np.nanstd(x, axis=0, keepdims=True), 1e-3)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(N_PCS, u.shape[1])
    return (u[:, :k] * s[:k]).astype(np.float32)

def local_scores(pcs: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    order = np.argsort(dist, axis=1)
    score = np.zeros(n, dtype=float)
    for i in range(n):
        bg = order[i, 8:40] if n >= 40 else order[i, 2:]
        if len(bg) < 4:
            continue
        med = np.median(pcs[bg], axis=0)
        mad = np.maximum(np.median(np.abs(pcs[bg] - med), axis=0), 0.2)
        z = np.abs((pcs[i] - med) / mad)
        score[i] = float(np.mean(np.sort(z)[-min(8, len(z)) :]))
    return score

def grow(seed: int, target: int, score: np.ndarray, adj: list[set[int]]) -> list[int]:
    chosen = {int(seed)}
    frontier = set(adj[int(seed)])
    while len(chosen) < target and frontier:
        best = max(frontier, key=lambda j: score[j])
        frontier.remove(best)
        chosen.add(best)
        for nb in adj[best]:
            if nb not in chosen:
                frontier.add(nb)
    return sorted(chosen)

def make_row(dataset_id: str, sample_id: str, spots: list[str], score: float, typ: str, params: str) -> dict[str, object]:
    cid = f"{dataset_id}__56C_dense__{typ}__{abs(hash((tuple(spots), params))) % 100000000:08d}"
    return {
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "candidate_id": cid,
        "proposal_id": cid,
        "proposal_type": typ,
        "member_spots": spots_to_text(spots),
        "n_spots": len(spots),
        "proposal_score": float(score),
        "proposal_params": params,
        "candidate_source": "56C_dense_local_primitives",
    }
