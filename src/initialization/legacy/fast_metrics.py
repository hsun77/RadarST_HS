from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import datetime,timezone
import hashlib,math,os,time,json
import numpy as np
import pandas as pd
from scipy import stats,sparse
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def connected_match_state(label_codes: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray):
    keep = label_codes[edge_i] == label_codes[edge_j]
    graph = coo_matrix((np.ones(int(keep.sum()) * 2, dtype=np.int8), (np.concatenate([edge_i[keep], edge_j[keep]]), np.concatenate([edge_j[keep], edge_i[keep]]))), shape=(len(label_codes), len(label_codes))).tocsr()
    (_n_components, component_codes) = connected_components(graph, directed=False, return_labels=True)
    return (component_codes, np.bincount(component_codes))

def best_match_fast(component_codes: np.ndarray, component_sizes: np.ndarray, truth_indices: np.ndarray):
    if len(truth_indices) == 0:
        return {'overlap': 0, 'region_size': 0, 'iou': 0.0}
    overlap_by_component = np.bincount(component_codes[truth_indices], minlength=len(component_sizes))
    candidate_ids = np.flatnonzero(overlap_by_component)
    overlaps = overlap_by_component[candidate_ids].astype(float)
    sizes = component_sizes[candidate_ids].astype(float)
    truth_size = float(len(truth_indices))
    ious = overlaps / np.maximum(1.0, truth_size + sizes - overlaps)
    order = np.lexsort((candidate_ids, sizes, -overlaps, -ious))
    chosen = int(order[0])
    return {'overlap': int(overlaps[chosen]), 'region_size': int(sizes[chosen]), 'iou': float(ious[chosen])}
