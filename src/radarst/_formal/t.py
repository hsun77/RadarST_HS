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


def blind_features(frame: pd.DataFrame, proposal_types: list[str]) -> pd.DataFrame:
    return D.feature_frame(frame, [], proposal_types)

def safety_mask(frame: pd.DataFrame, variant: str) -> np.ndarray:
    strict = D.safety_filter(frame)
    if variant == "strict":
        return strict
    if variant != "universal_local":
        raise ValueError(variant)
    local = (
        pd.to_numeric(frame.n_spots, errors="coerce").between(1, 10)
        & pd.to_numeric(frame.touched_upstream_components, errors="coerce").le(1)
        & pd.to_numeric(frame.candidate_label_purity, errors="coerce").ge(0.50)
        & pd.to_numeric(frame.single_boundary_percentile, errors="coerce").ge(0.75)
        & pd.to_numeric(frame.proposal_percentile, errors="coerce").ge(0.50)
    ).to_numpy(bool)
    return strict | local

def accept(frame, scores, spot_ids, method, safety_variant, budget):
    work = frame.copy()
    work["selection_score"] = scores
    work = work.loc[safety_mask(work, safety_variant)].copy()
    work = work.sort_values(
        ["selection_score", "single_boundary_percentile", "proposal_percentile", "candidate_key"],
        ascending=[False, False, False, True],
    )
    spot_set = set(map(str, spot_ids))
    maximum_modified = int(math.floor(len(spot_ids) * budget))
    touched, assignment, rows = set(), {}, []
    variant = f"method_blind_{safety_variant}_{budget:.3f}"
    for row in work.itertuples(index=False):
        members = [spot for spot in B67.parse_spots(row.canonical_member_spots) if spot in spot_set]
        member_set = set(members)
        if not member_set or member_set & touched:
            continue
        if len(touched) + len(member_set) > maximum_modified:
            continue
        label = f"RadarST_SINGLE_BLIND__{method}__{row.candidate_id}"
        assignment.update({spot: label for spot in member_set})
        touched.update(member_set)
        rows.append({
            "dataset_id": row.dataset_id,
            "sample_id": row.sample_id,
            "method": method,
            "variant": variant,
            "candidate_id": row.candidate_id,
            "candidate_key": row.candidate_key,
            "proposal_type": row.proposal_type,
            "member_spots": row.canonical_member_spots,
            "n_spots": len(member_set),
            "selection_score": row.selection_score,
            "modified_spots_after_acceptance": len(touched),
        })
        if len(touched) >= maximum_modified:
            break
    return pd.DataFrame(rows), assignment, variant
