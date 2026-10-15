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


def is_connected(indices: set[int], adjacency: list[list[int]]) -> bool:
    if not indices:
        return False
    start = next(iter(indices))
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in indices and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(indices)

def residual_component_count(
    component: set[int],
    removed: set[int],
    adjacency: list[list[int]],
) -> int:
    residual_size = len(component) - len(removed)
    if residual_size <= 0:
        return 0
    boundary = {
        neighbor
        for node in removed
        for neighbor in adjacency[node]
        if neighbor in component and neighbor not in removed
    }
    if not boundary:
        return 1

    unseen_boundary = set(boundary)
    n_components = 0
    while unseen_boundary:
        n_components += 1
        start = next(iter(unseen_boundary))
        seen = {start}
        queue = deque([start])
        unseen_boundary.discard(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in removed or neighbor not in component or neighbor in seen:
                    continue
                seen.add(neighbor)
                unseen_boundary.discard(neighbor)
                queue.append(neighbor)
            if not unseen_boundary:
                return n_components
    return n_components

def method_component_index(
    base: pd.Series,
    context: dict[str, object],
) -> tuple[dict[str, set[int]], list[str], np.ndarray]:
    spot_ids = list(map(str, context["spot_ids"]))
    spot_to_index = {spot: index for index, spot in enumerate(spot_ids)}
    component_sets, spot_to_component, _ = B67.label_connected_components(
        base, spot_ids, context["adjacency"]
    )
    components = {
        component_id: {spot_to_index[spot] for spot in spots}
        for component_id, spots in component_sets.items()
    }
    component_id_by_index = [spot_to_component[spot] for spot in spot_ids]
    labels = base.reindex(spot_ids).fillna("missing_pred").astype(str).to_numpy()
    return components, component_id_by_index, labels
