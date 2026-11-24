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


def adjacency_at_radius(coords: np.ndarray, radius: float) -> list[list[int]]:
    tree = cKDTree(coords)
    adjacency = [set() for _ in range(len(coords))]
    for (i, j) in tree.query_pairs(r=float(radius)):
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))
    return [sorted(neighbors) for neighbors in adjacency]

def infer_radius(coords: np.ndarray) -> float:
    if len(coords) <= 1:
        return 1.0
    tree = cKDTree(coords)
    (distances, _) = tree.query(coords, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest) * 1.25) if len(nearest) else 1.0
