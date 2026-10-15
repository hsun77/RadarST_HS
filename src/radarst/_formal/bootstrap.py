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


SMALL_MAX = 10

def primary_hierarchy(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    small = events.loc[events.component_size_spots.between(1, SMALL_MAX)].copy()
    pair = (
        small.groupby(["cohort", "dataset_id", "sample_id", "method"], as_index=False)
        .agg(
            n_events=("truth_component_id", "size"),
            base_best_IoU=("base_best_IoU", "mean"),
            after_best_IoU=("after_best_IoU", "mean"),
            observed_gain=("delta_best_IoU", "mean"),
            improved_events=("delta_best_IoU", lambda x: int((x > 1e-12).sum())),
            unchanged_events=("delta_best_IoU", lambda x: int((x.abs() <= 1e-12).sum())),
            decreased_events=("delta_best_IoU", lambda x: int((x < -1e-12).sum())),
        )
    )
    sample = (
        pair.groupby(["cohort", "dataset_id", "sample_id"], as_index=False)
        .agg(
            n_methods=("method", "nunique"),
            base_best_IoU=("base_best_IoU", "mean"),
            after_best_IoU=("after_best_IoU", "mean"),
            observed_gain=("observed_gain", "mean"),
        )
    )
    return pair, sample

def bootstrap_ci(sample: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for offset, (cohort, group) in enumerate(sample.groupby("cohort", sort=False)):
        values = group.observed_gain.to_numpy(float)
        rng = np.random.default_rng(seed + offset * 100003)
        draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "cohort": cohort,
                "n_samples": len(values),
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": seed + offset * 100003,
                "base_best_IoU": group.base_best_IoU.mean(),
                "after_best_IoU": group.after_best_IoU.mean(),
                "gain": values.mean(),
                "ci95_lower": np.quantile(draws, 0.025),
                "ci95_upper": np.quantile(draws, 0.975),
                "bootstrap_median": np.median(draws),
                "bootstrap_sd": draws.std(ddof=1),
            }
        )
    return pd.DataFrame(rows)
