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


def parse_spots(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    return [x for x in text.split(";") if x] if text else []

def load_prediction(path: Path, spot_index: pd.Index, barcode_col: str | None = None, pred_col: str | None = None) -> pd.Series:
    df = pd.read_csv(path, dtype=str).fillna("")
    if barcode_col not in df.columns:
        barcode_col = "spot_id" if "spot_id" in df.columns else ("barcode" if "barcode" in df.columns else df.columns[0])
    if pred_col not in df.columns:
        pred_col = "pred_label" if "pred_label" in df.columns else [c for c in df.columns if c != barcode_col][-1]
    dedup = df.drop_duplicates(barcode_col).set_index(barcode_col)[pred_col].astype(str)
    aligned = dedup.reindex(spot_index)
    if float(aligned.notna().mean()) < 0.50 and len(df) == len(spot_index):
        return pd.Series(df[pred_col].astype(str).to_numpy(), index=spot_index)
    return aligned.fillna("missing_pred").astype(str)

def label_connected_components(
    labels: pd.Series,
    spot_ids: list[str],
    adjacency: list[list[int]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    label_values = labels.reindex(spot_ids).fillna("missing_pred").astype(str).to_numpy()
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(label_values):
        by_label[str(label)].append(idx)
    component_sets: dict[str, set[str]] = {}
    spot_to_component: dict[str, str] = {}
    label_to_components: dict[str, list[str]] = defaultdict(list)
    for label, indices in by_label.items():
        allowed = set(indices)
        seen: set[int] = set()
        component_number = 0
        for start in indices:
            if start in seen:
                continue
            component_number += 1
            queue = deque([start])
            seen.add(start)
            members: list[int] = []
            while queue:
                node = queue.popleft()
                members.append(node)
                for neighbor in adjacency[node]:
                    if neighbor in allowed and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            component_id = f"{label}::cc{component_number:04d}"
            member_spots = {spot_ids[i] for i in members}
            component_sets[component_id] = member_spots
            label_to_components[label].append(component_id)
            for spot in member_spots:
                spot_to_component[spot] = component_id
    return component_sets, spot_to_component, dict(label_to_components)
