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
ROOT = Path('.')
table = None
native_reads = []
original_feature_pcs = None

def expression_only_feature_pcs(path, kind, spots):
    if kind != 'h5ad':
        return original_feature_pcs(path, kind, spots)
    import h5py, anndata as ad, scanpy as sc
    declared = {str((ROOT / s).resolve()) for s in table.expression}
    assert str(Path(path).resolve()) in declared
    with h5py.File(path, 'r') as h:
        obskey = h['obs'].attrs['_index']
        varkey = h['var'].attrs['_index']
        x = ad.io.read_elem(h['X'])
        obs = ad.io.read_elem(h['obs'][obskey])
        var = ad.io.read_elem(h['var'][varkey])
    native_reads.append({'path': str(path), 'elements_read': ['X', 'obs/' + str(obskey), 'var/' + str(varkey)], 'obs_pathology_values_read': False})
    data = ad.AnnData(x, obs=pd.DataFrame(index=list(map(str, obs))), var=pd.DataFrame(index=list(map(str, var))))
    data.var_names_make_unique()
    data = data[spots].copy()
    sc.pp.normalize_total(data, target_sum=10000.0)
    sc.pp.log1p(data)
    if data.n_vars > 3000:
        sc.pp.highly_variable_genes(data, n_top_genes=3000, flavor='seurat', subset=True)
    sc.pp.pca(data, n_comps=max(2, min(30, data.n_obs - 1, data.n_vars - 1)), zero_center=True, random_state=20260805)
    pcs = np.asarray(data.obsm['X_pca'], dtype=np.float32)
    med = np.median(pcs, axis=0)
    return ((pcs - med) / np.maximum(1.4826 * np.median(np.abs(pcs - med), axis=0), 0.001)).astype(np.float32)
