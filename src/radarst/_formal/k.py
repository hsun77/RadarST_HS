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


def predict(frame: pd.DataFrame, models: dict, proposal_types: list[str]) -> pd.DataFrame:
    x = G.feature_frame(frame, proposal_types)
    gain_parts = np.column_stack(
        [model.predict_proba(x)[:, 1] for model in models["positive_models"]]
    )
    rank_parts = np.column_stack(
        [np.clip(model.predict(x), 0.0, 1.0) for model in models["rank_models"]]
    )
    harm_parts = np.column_stack(
        [model.predict_proba(x)[:, 1] for model in models["harm_models"]]
    )
    gain_probability = gain_parts.mean(axis=1)
    gain_rank = rank_parts.mean(axis=1)
    harm_probability = harm_parts.mean(axis=1)
    expected_gain_rank = gain_probability * gain_rank
    utility_rank = expected_gain_rank * (1.0 - harm_probability)
    conservative = np.clip(
        (gain_parts.min(axis=1) * rank_parts.min(axis=1))
        * (1.0 - harm_parts.max(axis=1)),
        0.0,
        1.0,
    )
    signed_rank = expected_gain_rank - harm_probability
    return pd.DataFrame(
        {
            "gain_probability": gain_probability,
            "gain_probability_std": gain_parts.std(axis=1),
            "predicted_positive_rank": gain_rank,
            "predicted_positive_rank_std": rank_parts.std(axis=1),
            "harm_probability": harm_probability,
            "harm_probability_max": harm_parts.max(axis=1),
            "score_expected_gain_rank": expected_gain_rank,
            "score_utility_rank": utility_rank,
            "score_conservative_rank": conservative,
            "score_signed_rank": signed_rank,
        }
    )
