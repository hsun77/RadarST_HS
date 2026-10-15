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


def raw_sensor_evidence(
    candidates: pd.DataFrame,
    context: dict[str, object],
    sensors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[set[int]], Counter[str]]:
    spot_ids = list(map(str, context["spot_ids"]))
    coords = np.asarray(context["coords"], dtype=float)
    spot_to_index = {spot: index for index, spot in enumerate(spot_ids)}
    nearest = NearestNeighbors(n_neighbors=min(9, len(spots := spot_ids))).fit(coords)
    neighbor_rows = nearest.kneighbors(coords, return_distance=False)
    neighbors = [list(map(int, row[1:])) for row in neighbor_rows]
    cluster_sizes = []
    for method_index in range(sensors.shape[1]):
        values, counts = np.unique(sensors[:, method_index], return_counts=True)
        cluster_sizes.append(dict(zip(map(int, values), map(int, counts))))
    rare_cut = max(10, int(np.ceil(0.02 * len(spot_ids))))
    # Keep float64 parity with the frozen 57D/72K1 implementations. Casting the
    # boundary fractions to float32 perturbs percentile ties and can change the
    # downstream atomic writeback order.
    boundaries = np.zeros((len(candidates), sensors.shape[1]), dtype=float)
    rare = np.zeros_like(boundaries)
    candidate_indices: list[set[int]] = []
    audit: Counter[str] = Counter()

    for row_number, row in enumerate(candidates.itertuples(index=False)):
        members = list(dict.fromkeys(B67.parse_spots(row.member_spots)))
        if not members or any(spot not in spot_to_index for spot in members):
            candidate_indices.append(set())
            audit["invalid_or_empty_spots"] += 1
            continue
        indices = {spot_to_index[spot] for spot in members}
        candidate_indices.append(indices)
        background = sorted(
            {
                neighbor
                for index in indices
                for neighbor in neighbors[index]
                if neighbor not in indices
            }
        )
        if not background:
            audit["empty_local_background"] += 1
            continue
        cidx = np.fromiter(sorted(indices), dtype=int)
        bidx = np.asarray(background, dtype=int)
        for method_index in range(sensors.shape[1]):
            candidate_labels = sensors[cidx, method_index]
            background_labels = sensors[bidx, method_index]
            boundaries[row_number, method_index] = float(
                (candidate_labels[:, None] != background_labels[None, :]).mean()
            )
            values, counts = np.unique(candidate_labels, return_counts=True)
            dominant = int(values[np.argmax(counts)])
            rare[row_number, method_index] = float(
                cluster_sizes[method_index][dominant] <= rare_cut
            )
    audit["raw_candidates"] = len(candidates)
    return boundaries, rare, candidate_indices, audit

def topology_safety(
    union_ids: set[str],
    evidence_reference: pd.DataFrame,
    candidate_indices: list[set[int]],
    upstream: pd.Series,
    context: dict[str, object],
) -> dict[str, dict[str, object]]:
    spot_ids = list(map(str, context["spot_ids"]))
    adjacency = context["adjacency"]
    components, component_id_by_index, labels = B71.method_component_index(upstream, context)
    rows = evidence_reference.reset_index(drop=True)
    result: dict[str, dict[str, object]] = {}
    for row_number, row in enumerate(rows.itertuples(index=False)):
        candidate_key = str(row.candidate_row_key)
        candidate_id = str(row.candidate_id)
        if candidate_key not in union_ids:
            continue
        indices = candidate_indices[row_number]
        if not indices or not B71.is_connected(indices, adjacency):
            continue
        counts = Counter(component_id_by_index[index] for index in indices)
        if any(count == len(components[component_id]) for component_id, count in counts.items()):
            continue
        dominant_fraction = max(counts.values()) / len(indices)
        cross_component_risk = 1.0 - dominant_fraction
        erosion_risk = max(
            count / len(components[component_id]) for component_id, count in counts.items()
        )
        outside_labels = [
            str(labels[neighbor])
            for index in indices
            for neighbor in adjacency[index]
            if neighbor not in indices
        ]
        if outside_labels:
            dominant_boundary = Counter(outside_labels).most_common(1)[0][1] / len(outside_labels)
            boundary_leakage_risk = 1.0 - dominant_boundary
        else:
            dominant_boundary = 0.0
            boundary_leakage_risk = 1.0
        residual_extra_components = 0
        for component_id in counts:
            component = components[component_id]
            removed = indices & component
            residual_extra_components += max(
                0, B71.residual_component_count(component, removed, adjacency) - 1
            )
        fragmentation_risk = min(1.0, residual_extra_components / 2.0)
        safety_score = float(
            np.clip(
                1.0
                - np.mean(
                    [
                        fragmentation_risk,
                        cross_component_risk,
                        boundary_leakage_risk,
                        erosion_risk,
                    ]
                ),
                0.0,
                1.0,
            )
        )
        result[candidate_key] = {
            "indices": indices,
            "member_spots": str(row.member_spots),
            "candidate_id": candidate_id,
            "candidate_row_key": candidate_key,
            "proposal_type": str(row.proposal_type),
            "n_spots": len(indices),
            "fragmentation_risk": fragmentation_risk,
            "residual_extra_components": residual_extra_components,
            "cross_component_risk": cross_component_risk,
            "boundary_leakage_risk": boundary_leakage_risk,
            "dominant_boundary_label_fraction": dominant_boundary,
            "erosion_risk": erosion_risk,
            "safety_score": safety_score,
            "touched_upstream_components": len(counts),
        }
    return result
