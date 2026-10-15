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


def blind_features(frame: pd.DataFrame, proposal_types: list[str]) -> pd.DataFrame:
    return D.feature_frame(frame, [], proposal_types)

def fit_model(training: pd.DataFrame, proposal_types: list[str]):
    model = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=D.RANDOM_SEED,
    )
    y = training.teacher_E7_selected.to_numpy(int)
    model.fit(blind_features(training, proposal_types), y, sample_weight=D.balanced_weights(y))
    return model
