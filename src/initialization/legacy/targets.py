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


SEED = 20260804

def component_maps(components: pd.DataFrame):
    comp_sets = {}
    comp_sizes = {}
    spot_to_components = defaultdict(list)
    component_sizes = pd.to_numeric(components.component_size_spots, errors='coerce')
    for row in components[component_sizes.between(1, 10)].itertuples(index=False):
        cid = str(row.truth_component_id)
        raw_spots = row.truth_spots
        if isinstance(raw_spots, (list, tuple, set, np.ndarray, pd.Series)):
            spots = {str(spot) for spot in raw_spots if str(spot)}
        else:
            spots = set(B67.parse_spots(raw_spots))
        comp_sets[cid] = spots
        comp_sizes[cid] = len(spots)
        for spot in spots:
            spot_to_components[spot].append(cid)
    return (comp_sets, comp_sizes, spot_to_components)

def candidate_truth_overlap(frame, comp_sets, spot_to_components):
    (best_iou, best_cid) = ([], [])
    for value in frame.canonical_member_spots:
        members = set(B67.parse_spots(value))
        candidate_components = set()
        for spot in members:
            candidate_components.update(spot_to_components.get(spot, []))
        best = (0.0, '')
        # Clean correction: stable tie resolution, independent of hash seed.
        # Maximum IoU remains the target criterion; equal maxima use component ID.
        for cid in sorted(candidate_components):
            truth = comp_sets[cid]
            overlap = len(members & truth)
            iou = overlap / max(1, len(members | truth))
            if iou > best[0]:
                best = (iou, cid)
        best_iou.append(best[0])
        best_cid.append(best[1])
    return (np.asarray(best_iou, dtype=float), np.asarray(best_cid, dtype=object))

def training_sample(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame[frame.utility_target.gt(0)].copy()
    zero = frame[frame.utility_target.le(0)].copy()
    selected = set(positive.index)
    for column in ('proposal_percentile', 'single_boundary_percentile', 'safety_score'):
        selected.update(zero.nlargest(min(180, len(zero)), column).index)
    if len(zero):
        selected.update(zero.sample(min(300, len(zero)), random_state=SEED).index)
    return frame.loc[sorted(selected)].copy()
