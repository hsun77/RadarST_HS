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


def candidate_records(frame: pd.DataFrame, spot_ids: list[str]):
    valid = set(map(str, spot_ids))
    records = []
    for index, row in enumerate(frame.itertuples(index=False)):
        members = frozenset(
            spot
            for spot in B67.parse_spots(row.canonical_member_spots)
            if spot in valid
        )
        if not members:
            continue
        records.append(
            {
                "index": index,
                "candidate_id": str(row.candidate_id),
                "candidate_key": str(row.candidate_key),
                "proposal_type": str(row.proposal_type),
                "n_spots": len(members),
                "members": members,
            }
        )
    return records

def eligibility_masks(frame: pd.DataFrame):
    non = ~frame.proposal_type.astype(str).eq("singleton")
    route = E.D22.U.adaptive_safety(frame.loc[non].copy())
    core = np.zeros(len(frame), dtype=bool)
    core[non.to_numpy()] = T.safety_mask(frame.loc[non], route)
    singleton = frame.proposal_type.astype(str).eq("singleton").to_numpy(bool)
    singleton &= E.D22.singleton_gate(frame, "strict_only")
    tail = non.to_numpy(bool) & T.safety_mask(frame, "strict")
    return core, singleton, tail, route
