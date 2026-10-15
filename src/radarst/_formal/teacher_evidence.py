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


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(float(numeric.median()) if numeric.notna().any() else 0.0)
    return numeric.rank(method="average", pct=True, ascending=True)

def prepare_candidate_evidence(protocol: dict[str, object]) -> pd.DataFrame:
    config = protocol["candidate_input"]
    path = Path(config["path"])
    header = pd.read_csv(path, nrows=0).columns.tolist()
    forbidden = set(config["truth_derived_columns_forbidden"])
    allowed = [column for column in config["allowed_columns"] if column in header]
    if forbidden.intersection(allowed):
        raise RuntimeError("Truth-derived candidate columns entered the selector whitelist")
    required = {
        "dataset_id",
        "sample_id",
        "candidate_id",
        "proposal_type",
        "member_spots",
        "proposal_score",
        "sensor_boundary_mean",
        "sensor_island_support_frac",
        "sensor_rare_support_frac",
    }
    if not required.issubset(allowed):
        raise RuntimeError(f"Missing safe candidate columns: {sorted(required - set(allowed))}")
    candidates = pd.read_csv(path, usecols=allowed, low_memory=False)
    candidates["member_spots"] = candidates["member_spots"].fillna("").astype(str)
    parsed_sizes = candidates["member_spots"].map(lambda value: len(set(B67.parse_spots(value))))
    candidates["n_spots"] = pd.to_numeric(candidates.get("n_spots", parsed_sizes), errors="coerce")
    candidates["n_spots"] = candidates["n_spots"].fillna(parsed_sizes).astype(int)
    for column in [
        "proposal_score",
        "sensor_boundary_mean",
        "sensor_island_support_frac",
        "sensor_rare_support_frac",
    ]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
    candidates["proposal_family"] = candidates["proposal_type"].fillna("unknown").astype(str)
    candidates["is_singleton"] = candidates["n_spots"].eq(1)
    candidates["compactness_raw"] = 1.0 / np.log1p(1.0 + candidates["n_spots"].clip(lower=1))

    group = candidates.groupby(["dataset_id", "proposal_family"], sort=False)
    candidates["boundary_percentile"] = group["sensor_boundary_mean"].transform(percentile_rank)
    candidates["proposal_percentile"] = group["proposal_score"].transform(percentile_rank)
    candidates["compactness_percentile"] = group["compactness_raw"].transform(percentile_rank)
    candidates["evidence_score"] = (
        0.40 * candidates["boundary_percentile"]
        + 0.25 * candidates["proposal_percentile"]
        + 0.20 * candidates["sensor_island_support_frac"].clip(0, 1)
        + 0.10 * candidates["sensor_rare_support_frac"].clip(0, 1)
        + 0.05 * candidates["compactness_percentile"]
    )

    gates = protocol["candidate_gates"]
    denominator = int(gates["sensor_method_denominator"])
    candidates["island_support_methods"] = np.rint(
        candidates["sensor_island_support_frac"] * denominator
    ).astype(int)
    candidates["rare_support_methods"] = np.rint(
        candidates["sensor_rare_support_frac"] * denominator
    ).astype(int)
    singleton = gates["singleton"]
    multispot = gates["multispot"]
    singleton_pass = (
        candidates["is_singleton"]
        & candidates["boundary_percentile"].ge(float(singleton["boundary_percentile_min"]))
        & candidates["proposal_percentile"].ge(float(singleton["proposal_percentile_min"]))
        & (
            candidates["island_support_methods"].ge(int(singleton["minimum_island_support_methods"]))
            | candidates["rare_support_methods"].ge(
                int(singleton["minimum_rare_support_methods_alternative"])
            )
        )
    )
    multispot_pass = (
        ~candidates["is_singleton"]
        & candidates["boundary_percentile"].ge(float(multispot["boundary_percentile_min"]))
        & candidates["proposal_percentile"].ge(float(multispot["proposal_percentile_min"]))
        & (
            candidates["island_support_methods"].ge(int(multispot["minimum_island_support_methods"]))
            | candidates["rare_support_methods"].ge(
                int(multispot["minimum_rare_support_methods_alternative"])
            )
        )
    )
    candidates["passes_evidence_gate"] = singleton_pass | multispot_pass
    candidates = candidates.sort_values(
        [
            "dataset_id",
            "evidence_score",
            "sensor_boundary_mean",
            "proposal_score",
            "candidate_id",
        ],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["global_evidence_rank"] = (
        candidates.groupby("dataset_id", sort=False).cumcount() + 1
    )
    return candidates
