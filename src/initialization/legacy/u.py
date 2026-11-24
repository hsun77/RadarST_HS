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


def adaptive_safety(frame: pd.DataFrame) -> str:
    return 'strict' if int(T.safety_mask(frame, 'strict').sum()) > 0 else 'universal_local'
