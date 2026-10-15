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


SEED = 20260804

def fit_signed_heads(training, proposal_types):
    X = D.feature_frame(training, [], proposal_types)
    y = training.signed_utility_target.to_numpy(float)
    positive = y > 0
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
    )
    classifier.fit(X, positive.astype(int), sample_weight=D.balanced_weights(positive.astype(int)))
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=SEED,
        loss="squared_error",
    )
    regressor.fit(X, y)
    return classifier, regressor

def signed_score(frame, classifier, regressor, proposal_types, harm_weight):
    X = D.feature_frame(frame, [], proposal_types)
    probability = classifier.predict_proba(X)[:, 1]
    signed = np.clip(regressor.predict(X), -1.0, 1.0)
    return (
        probability * np.clip(signed, 0.0, 1.0)
        - harm_weight * (1.0 - probability) * np.clip(-signed, 0.0, 1.0)
    )
