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


def signed_score(frame, classifier, regressor, proposal_types, harm_weight):
    X = D.feature_frame(frame, [], proposal_types)
    probability = classifier.predict_proba(X)[:, 1]
    signed = np.clip(regressor.predict(X), -1.0, 1.0)
    return (
        probability * np.clip(signed, 0.0, 1.0)
        - harm_weight * (1.0 - probability) * np.clip(-signed, 0.0, 1.0)
    )
