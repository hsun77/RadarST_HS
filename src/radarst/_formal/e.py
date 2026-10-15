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


GATE = "strict_only"

SINGLETON_BUDGET = 0.01

def add_utility_tail(frame, teacher_scores, utility_scores, spot_ids, method, extra_budget):
    accepted, assignment, _route = D22.accept_additive(
        frame, teacher_scores, spot_ids, method, GATE, SINGLETON_BUDGET
    )
    if len(accepted):
        accepted = accepted.copy()
        accepted["stage"] = "protected_v11"
    valid_spots = set(map(str, spot_ids))
    touched = set(assignment)
    maximum_extra = int(math.floor(len(spot_ids) * extra_budget))
    extra_spots = set()

    work = frame.loc[~frame.proposal_type.astype(str).eq("singleton")].copy()
    work["selection_score"] = utility_scores[
        (~frame.proposal_type.astype(str).eq("singleton")).to_numpy()
    ]
    work = work.loc[T.safety_mask(work, "strict")].copy()
    work = work.sort_values(
        ["selection_score", "single_boundary_percentile", "proposal_percentile", "candidate_key"],
        ascending=[False, False, False, True],
    )
    rows = []
    for row in work.itertuples(index=False):
        members = {s for s in B67.parse_spots(row.canonical_member_spots) if s in valid_spots}
        if not members or members & touched:
            continue
        if len(extra_spots) + len(members) > maximum_extra:
            continue
        label = f"RadarST_SINGLE_v12_utility_tail__{method}__{row.candidate_id}"
        assignment.update({spot: label for spot in members})
        touched.update(members)
        extra_spots.update(members)
        rows.append(
            {
                "dataset_id": row.dataset_id,
                "sample_id": row.sample_id,
                "method": method,
                "candidate_id": row.candidate_id,
                "candidate_key": row.candidate_key,
                "proposal_type": row.proposal_type,
                "member_spots": row.canonical_member_spots,
                "n_spots": len(members),
                "selection_score": row.selection_score,
                "stage": "additive_signed_utility_tail",
            }
        )
        if len(extra_spots) >= maximum_extra:
            break
    if rows:
        accepted = pd.concat([accepted, pd.DataFrame(rows)], ignore_index=True)
    return accepted, assignment
