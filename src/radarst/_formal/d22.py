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


BASE_BUDGET = 0.125

def singleton_gate(frame: pd.DataFrame, gate: str) -> np.ndarray:
    strict = T.safety_mask(frame, "strict")
    if gate == "strict_only":
        return strict
    q = 0.75 if gate == "q75_support" else 0.90
    support = (
        pd.to_numeric(frame.proposal_percentile, errors="coerce").fillna(0).ge(q)
        & pd.to_numeric(frame.single_boundary_percentile, errors="coerce").fillna(0).ge(q)
        & pd.to_numeric(frame.touched_upstream_components, errors="coerce").fillna(99).le(1)
    ).to_numpy(bool)
    return strict & support

def accept_additive(frame, scores, spot_ids, method, gate, additive_budget):
    non = ~frame.proposal_type.astype(str).eq("singleton")
    non_frame = frame.loc[non].copy()
    non_scores = scores[non.to_numpy()]
    safety_route = U.adaptive_safety(non_frame)
    base_accepted, assignment, _variant = T.accept(
        non_frame, non_scores, spot_ids, method, safety_route, BASE_BUDGET
    )
    if len(base_accepted):
        base_accepted = base_accepted.copy()
        base_accepted["stage"] = "protected_non_singleton"

    spot_set = set(map(str, spot_ids))
    touched = set(assignment)
    maximum_total = int(math.floor(len(spot_ids) * (BASE_BUDGET + additive_budget)))
    maximum_singletons = int(math.floor(len(spot_ids) * additive_budget))
    singleton_count = 0
    singleton_rows = []

    singleton_mask = frame.proposal_type.astype(str).eq("singleton").to_numpy(bool)
    singleton_mask &= singleton_gate(frame, gate)
    single = frame.loc[singleton_mask].copy()
    single["selection_score"] = scores[singleton_mask]
    single = single.sort_values(
        ["selection_score", "single_boundary_percentile", "proposal_percentile", "candidate_key"],
        ascending=[False, False, False, True],
    )
    for row in single.itertuples(index=False):
        members = [s for s in B67.parse_spots(row.canonical_member_spots) if s in spot_set]
        member_set = set(members)
        if not member_set or member_set & touched:
            continue
        if singleton_count + len(member_set) > maximum_singletons:
            continue
        if len(touched) + len(member_set) > maximum_total:
            continue
        label = f"RadarST_SINGLE_v11_add__{method}__{row.candidate_id}"
        assignment.update({spot: label for spot in member_set})
        touched.update(member_set)
        singleton_count += len(member_set)
        singleton_rows.append(
            {
                "dataset_id": row.dataset_id,
                "sample_id": row.sample_id,
                "method": method,
                "candidate_id": row.candidate_id,
                "candidate_key": row.candidate_key,
                "proposal_type": row.proposal_type,
                "member_spots": row.canonical_member_spots,
                "n_spots": len(member_set),
                "selection_score": row.selection_score,
                "stage": "additive_singleton",
            }
        )
        if singleton_count >= maximum_singletons or len(touched) >= maximum_total:
            break
    singleton_df = pd.DataFrame(singleton_rows)
    accepted = pd.concat(
        [x for x in (base_accepted, singleton_df) if len(x)], ignore_index=True
    ) if len(base_accepted) or len(singleton_df) else pd.DataFrame()
    return accepted, assignment, safety_route
