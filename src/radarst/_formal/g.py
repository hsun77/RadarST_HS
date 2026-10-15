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
