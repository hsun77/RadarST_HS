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


SEED = 20260805

def group_rank_target(frame: pd.DataFrame, signed: np.ndarray) -> np.ndarray:
    values = pd.Series(np.maximum(signed, 0.0), index=frame.index)
    groups = [frame.dataset_id.astype(str), frame.method.astype(str)]
    rank = values.groupby(groups).rank(method='average', pct=True).to_numpy(float)
    rank[signed <= 0] = 0.0
    return rank

def class_group_weights(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    groups = frame.dataset_id.astype(str) + '||' + frame.method.astype(str)
    counts = groups.value_counts()
    weights = groups.map(lambda key: 1.0 / counts[key]).to_numpy(float)
    for label in np.unique(labels):
        mask = labels == label
        if mask.any():
            weights[mask] *= 1.0 / max(weights[mask].sum(), 1e-12)
    return weights * len(weights) / max(weights.sum(), 1e-12)

def fit_models(training: pd.DataFrame, proposal_types: list[str]) -> dict:
    x = G.feature_frame(training, proposal_types)
    signed = pd.to_numeric(training.signed_utility_target, errors='coerce').fillna(0).to_numpy(float)
    positive = signed > 0
    harm = signed < 0
    rank_target = group_rank_target(training, signed)
    classifier_configs = ((0.045, 260, 15, 55, 4.0), (0.05, 240, 31, 45, 3.0), (0.04, 280, 63, 35, 5.0))
    positive_models = []
    for (index, (lr, iterations, leaves, minimum, l2)) in enumerate(classifier_configs):
        model = HistGradientBoostingClassifier(learning_rate=lr, max_iter=iterations, max_leaf_nodes=leaves, min_samples_leaf=minimum, l2_regularization=l2, random_state=SEED + index)
        model.fit(x, positive.astype(np.int8), sample_weight=class_group_weights(training, positive))
        positive_models.append(model)
    rank_models = []
    positive_frame = training.loc[positive]
    x_positive = x.loc[positive]
    for (index, (lr, iterations, leaves, minimum, l2)) in enumerate(classifier_configs):
        model = HistGradientBoostingRegressor(learning_rate=lr, max_iter=iterations, max_leaf_nodes=leaves, min_samples_leaf=minimum, l2_regularization=l2, loss='absolute_error' if index != 1 else 'squared_error', random_state=SEED + 10 + index)
        model.fit(x_positive, rank_target[positive], sample_weight=class_group_weights(positive_frame, np.ones(len(positive_frame), dtype=np.int8)))
        rank_models.append(model)
    harm_models = G.fit_ensemble(training, proposal_types)
    return {'positive_models': positive_models, 'rank_models': rank_models, 'harm_models': harm_models, 'proposal_types': proposal_types, 'feature_columns': list(x.columns)}

def predict(frame: pd.DataFrame, models: dict, proposal_types: list[str]) -> pd.DataFrame:
    x = G.feature_frame(frame, proposal_types)
    gain_parts = np.column_stack([model.predict_proba(x)[:, 1] for model in models['positive_models']])
    rank_parts = np.column_stack([np.clip(model.predict(x), 0.0, 1.0) for model in models['rank_models']])
    harm_parts = np.column_stack([model.predict_proba(x)[:, 1] for model in models['harm_models']])
    gain_probability = gain_parts.mean(axis=1)
    gain_rank = rank_parts.mean(axis=1)
    harm_probability = harm_parts.mean(axis=1)
    expected_gain_rank = gain_probability * gain_rank
    utility_rank = expected_gain_rank * (1.0 - harm_probability)
    conservative = np.clip(gain_parts.min(axis=1) * rank_parts.min(axis=1) * (1.0 - harm_parts.max(axis=1)), 0.0, 1.0)
    signed_rank = expected_gain_rank - harm_probability
    return pd.DataFrame({'gain_probability': gain_probability, 'gain_probability_std': gain_parts.std(axis=1), 'predicted_positive_rank': gain_rank, 'predicted_positive_rank_std': rank_parts.std(axis=1), 'harm_probability': harm_probability, 'harm_probability_max': harm_parts.max(axis=1), 'score_expected_gain_rank': expected_gain_rank, 'score_utility_rank': utility_rank, 'score_conservative_rank': conservative, 'score_signed_rank': signed_rank})
