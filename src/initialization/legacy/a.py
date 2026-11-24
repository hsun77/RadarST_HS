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


def engineered_features(frame: pd.DataFrame, proposal_types: list[str]) -> pd.DataFrame:
    x = D.feature_frame(frame, [], proposal_types).astype(np.float32)
    eps = np.float32(1e-06)
    x['contrast_x_boundary'] = x['score_per_spot_percentile'] * x['single_boundary_percentile']
    x['contrast_x_purity'] = x['score_per_spot_percentile'] * x['candidate_label_purity']
    x['boundary_x_purity'] = x['single_boundary_percentile'] * x['candidate_label_purity']
    x['safe_contrast'] = x['score_per_spot_percentile'] * x['safety_score']
    x['topology_risk_mean'] = x[['fragmentation_risk', 'cross_component_risk', 'boundary_leakage_risk', 'erosion_risk']].mean(axis=1)
    x['topology_risk_max'] = x[['fragmentation_risk', 'cross_component_risk', 'boundary_leakage_risk', 'erosion_risk']].max(axis=1)
    x['candidate_fraction_of_dominant'] = 1.0 / (np.expm1(x['log_dominant_cluster_to_candidate_ratio'].clip(0, 14)) + 1.0)
    x['compact_signal'] = x['proposal_percentile'] * x['compactness_percentile']
    x['boundary_per_log_size'] = x['single_boundary_percentile'] / (1.0 + x['log_n_spots'] + eps)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
