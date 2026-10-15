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


EPS = 1e-12

BASE_FEATURES = [
    "log_n_spots",
    "proposal_score",
    "score_per_spot",
    "proposal_percentile",
    "score_per_spot_percentile",
    "compactness_percentile",
    "multiscale_persistence_mean",
    "multiscale_persistence_min",
    "single_boundary_mean",
    "single_boundary_percentile",
    "single_island_support",
    "single_rare_cluster_support",
    "candidate_label_purity",
    "candidate_label_entropy",
    "log_dominant_cluster_to_candidate_ratio",
    "fragmentation_risk",
    "cross_component_risk",
    "boundary_leakage_risk",
    "erosion_risk",
    "safety_score",
    "touched_upstream_components",
]

def feature_frame(frame: pd.DataFrame, methods: list[str], proposal_types: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["log_n_spots"] = np.log1p(pd.to_numeric(frame["n_spots"], errors="coerce").fillna(0))
    result["log_dominant_cluster_to_candidate_ratio"] = np.log1p(
        pd.to_numeric(frame["dominant_cluster_to_candidate_ratio"], errors="coerce").fillna(1e6).clip(0, 1e6)
    )
    for column in BASE_FEATURES:
        if column in result:
            continue
        result[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for proposal_type in proposal_types:
        result[f"proposal::{proposal_type}"] = frame["proposal_type"].astype(str).eq(proposal_type).astype(np.int8)
    for method in methods:
        result[f"method::{method}"] = frame["method"].astype(str).eq(method).astype(np.int8)
    return result

def safety_filter(frame: pd.DataFrame) -> np.ndarray:
    nonfragmenting = frame["fragmentation_risk"].le(EPS) & frame["erosion_risk"].le(1 / 3 + EPS)
    safe_fragment = (
        frame["fragmentation_risk"].gt(EPS)
        & frame["erosion_risk"].le(0.05 + EPS)
        & frame["boundary_leakage_risk"].le(1 / 3 + EPS)
        & frame["single_boundary_percentile"].ge(0.65 - EPS)
    )
    return (frame["topology_valid"].astype(bool) & (nonfragmenting | safe_fragment)).to_numpy(bool)
