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


N_HVG = 1000

def read_expression(folder: Path, spot_order: list[str]) -> tuple[np.ndarray, list[str], pd.Index]:
    h5 = folder / 'filtered_feature_bc_matrix.h5'
    if not h5.exists():
        raise FileNotFoundError(h5)
    adata = sc.read_10x_h5(str(h5))
    adata.var_names_make_unique()
    obs = pd.Index(map(str, adata.obs_names))
    keep = [s for s in spot_order if s in obs]
    if not keep:
        raise RuntimeError(f'No expression barcodes align for {folder}')
    adata = adata[keep].copy()
    x = adata.X
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.astype(np.float32).tocsr()
    totals = np.asarray(x.sum(axis=1)).ravel()
    totals[totals <= 0] = 1.0
    x = x.multiply((10000.0 / totals)[:, None]).tocsr()
    x.data = np.log1p(x.data)
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean ** 2, 0)
    valid = np.where(mean > 0.01)[0]
    if len(valid) == 0:
        valid = np.arange(x.shape[1])
    top = valid[np.argsort(var[valid])[-min(N_HVG, len(valid)):]]
    dense = x[:, top].toarray().astype(np.float32)
    genes = list(map(str, adata.var_names[top]))
    return (dense, genes, pd.Index(keep))
