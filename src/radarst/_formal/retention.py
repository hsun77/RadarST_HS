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

def retained_assignment(accepted: pd.DataFrame, fraction: float, dataset_id: str, method: str):
    target = int(np.floor(float(accepted.n_spots.sum()) * fraction + EPS))
    ordered = accepted.sort_values(
        ["selection_score", "candidate_key"], ascending=[False, True]
    )
    chosen = []
    used = 0
    for row in ordered.itertuples(index=False):
        size = int(row.n_spots)
        if used + size > target:
            continue
        chosen.append(row)
        used += size
    assignment = {}
    for number, row in enumerate(chosen, start=1):
        label = f"RadarST_retained_{fraction:.3f}_{method}_{number}"
        for spot in str(row.member_spots).split(";"):
            if spot:
                assignment[spot] = label
    return assignment, len(chosen), used, target
