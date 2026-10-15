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


SEED = 20260805

def feature_frame(frame: pd.DataFrame, proposal_types: list[str]) -> pd.DataFrame:
    x = A.engineered_features(frame, proposal_types)
    added = [c for c in frame if c.startswith(("expr_", "graph_", "map_", "upstream_"))]
    group = [frame.dataset_id.astype(str), frame.method.astype(str)]
    for column in added:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x[column] = values.astype(np.float32)
        x[f"rank::{column}"] = values.groupby(group).rank(method="average", pct=True).astype(np.float32)
    x["harm_boundary_concordance"] = (
        x["upstream_boundary_cross_label_fraction"] * x["single_boundary_percentile"]
    )
    x["harm_fragmentation_context"] = x["fragmentation_risk"] * x["map_boundary_spot_fraction"]
    x["harm_erosion_context"] = x["erosion_risk"] * x["upstream_candidate_fraction_of_cluster"]
    x["harm_expression_boundary"] = (
        x["rank::expr_boundary_edge_mean"] * x["upstream_boundary_cross_label_fraction"]
    )
    x["harm_graph_cut"] = x["graph_conductance"] * x["upstream_member_boundary_fraction"]
    return x.astype(np.float32)

def sample_frame(frame: pd.DataFrame) -> pd.DataFrame:
    signed = pd.to_numeric(frame.signed_utility_target, errors="coerce").fillna(0)
    harm = frame[signed < 0]
    positive = frame[signed > 0]
    zero = frame[signed.eq(0)]
    selected = [harm]
    if len(positive):
        selected.append(positive.sample(min(1200, len(positive)), random_state=SEED))
    if len(zero):
        hard_zero = zero.nlargest(min(500, len(zero)), "single_boundary_percentile")
        random_zero = zero.drop(index=hard_zero.index).sample(
            min(700, max(0, len(zero) - len(hard_zero))), random_state=SEED
        )
        selected.extend([hard_zero, random_zero])
    return pd.concat(selected).drop_duplicates("candidate_row_key", keep="first")

def class_group_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    groups = frame.dataset_id.astype(str) + "||" + frame.method.astype(str)
    counts = groups.value_counts()
    w = groups.map(lambda key: 1.0 / counts[key]).to_numpy(float)
    for label in (False, True):
        mask = y == label
        if mask.any():
            w[mask] *= 0.5 / max(w[mask].sum(), 1e-12)
    return w * len(w) / max(w.sum(), 1e-12)

def fit_ensemble(training: pd.DataFrame, proposal_types: list[str]) -> list:
    x = feature_frame(training, proposal_types)
    signed = pd.to_numeric(training.signed_utility_target, errors="coerce").fillna(0).to_numpy(float)
    y = signed < 0
    weights = class_group_weights(training, y)
    configurations = (
        (0.055, 220, 31, 35, 2.0),
        (0.050, 240, 15, 60, 4.0),
        (0.045, 240, 63, 30, 3.0),
    )
    models = []
    for index, (lr, iterations, leaves, minimum, l2) in enumerate(configurations):
        model = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=iterations, max_leaf_nodes=leaves,
            min_samples_leaf=minimum, l2_regularization=l2, random_state=SEED + index,
        )
        model.fit(x, y.astype(np.int8), sample_weight=weights)
        models.append(model)
    return models
