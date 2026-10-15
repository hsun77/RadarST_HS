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


N_HVG = 1000

N_PCS = 20

CORE_TARGETS = [3, 4, 5, 8, 10]

def stable_candidate_id(sample_id: str, proposal_type: str, spots: list[str]) -> str:
    token = hashlib.sha1((proposal_type + "|" + ";".join(spots)).encode("utf-8")).hexdigest()[:14]
    return f"{sample_id}__72K1__{proposal_type}__{token}"

def read_expression(folder: Path, spot_order: list[str]) -> tuple[np.ndarray, pd.Index]:
    h5 = folder / "filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(str(h5))
    adata.var_names_make_unique()
    obs = pd.Index(map(str, adata.obs_names))
    keep = [spot for spot in spot_order if spot in obs]
    if not keep:
        raise RuntimeError(f"No expression barcodes align for {folder}")
    adata = adata[keep].copy()
    matrix = adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.astype(np.float32).tocsr()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals <= 0] = 1.0
    matrix = matrix.multiply((1e4 / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0)
    valid = np.where(mean > 0.01)[0]
    if len(valid) == 0:
        valid = np.arange(matrix.shape[1])
    top = valid[np.argsort(variance[valid])[-min(N_HVG, len(valid)) :]]
    return matrix[:, top].toarray().astype(np.float32), pd.Index(keep)

def pca_scores(expression: np.ndarray) -> np.ndarray:
    standardized = expression - np.nanmean(expression, axis=0, keepdims=True)
    standardized = standardized / np.maximum(np.nanstd(standardized, axis=0, keepdims=True), 1e-3)
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    n_components = min(N_PCS, u.shape[1])
    return (u[:, :n_components] * singular[:n_components]).astype(np.float32)

def local_scores(pcs: np.ndarray, coords: np.ndarray) -> np.ndarray:
    n_neighbors = min(40, len(coords))
    neighbor_order = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(
        coords, return_distance=False
    )
    result = np.zeros(len(coords), dtype=float)
    for index, row in enumerate(neighbor_order):
        background = row[8:40] if len(row) >= 40 else row[2:]
        if len(background) < 4:
            continue
        median = np.median(pcs[background], axis=0)
        mad = np.maximum(np.median(np.abs(pcs[background] - median), axis=0), 0.2)
        z_score = np.abs((pcs[index] - median) / mad)
        result[index] = float(np.mean(np.sort(z_score)[-min(8, len(z_score)) :]))
    return result

def radius_adjacency(coords: np.ndarray) -> list[set[int]]:
    nearest = NearestNeighbors(n_neighbors=min(2, len(coords))).fit(coords)
    distances, _ = nearest.kneighbors(coords)
    scale = float(np.nanmedian(distances[:, 1])) if distances.shape[1] > 1 else 1.0
    threshold = (scale if np.isfinite(scale) and scale > 0 else 1.0) * 1.55
    rows = nearest.radius_neighbors(coords, radius=threshold, return_distance=False)
    adjacency = [set(map(int, row)) - {index} for index, row in enumerate(rows)]
    for index, neighbors in enumerate(adjacency):
        for neighbor in list(neighbors):
            adjacency[neighbor].add(index)
    return adjacency

def grow(seed: int, target: int, scores: np.ndarray, adjacency: list[set[int]]) -> list[int]:
    chosen = {seed}
    frontier = set(adjacency[seed])
    while len(chosen) < target and frontier:
        best = max(frontier, key=lambda index: (scores[index], -index))
        frontier.remove(best)
        chosen.add(best)
        frontier.update(adjacency[best] - chosen)
    return sorted(chosen)

def generate_candidates(sample_id: str, spots: list[str], coords: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, list[set[int]]]:
    adjacency = radius_adjacency(coords)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    def append(indices: list[int], proposal_type: str, proposal_score: float) -> None:
        members = sorted(spots[index] for index in indices)
        key = tuple(members)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "dataset_id": sample_id,
                "sample_id": sample_id,
                "candidate_id": stable_candidate_id(sample_id, proposal_type, members),
                "proposal_type": proposal_type,
                "member_spots": ";".join(members),
                "n_spots": len(members),
                "proposal_score": float(proposal_score),
                "candidate_source": "72K1_expression_local_primitives_protocol_parity",
            }
        )

    for index in range(len(spots)):
        append([index], "singleton", scores[index])
        for neighbor in sorted(adjacency[index]):
            if index < neighbor:
                append([index, neighbor], "edge_pair", float((scores[index] + scores[neighbor]) / 2))
        for target in CORE_TARGETS:
            core = grow(index, target, scores, adjacency)
            if len(core) == target:
                append(core, f"core_{target}", float(np.mean(scores[core]) + scores[index]))
    return pd.DataFrame(rows), adjacency
