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

def load_prediction_map(row: pd.Series, spot_ids: list[str]) -> pd.Series:
    path = Path(patch_path_text(row["pred_path"]))
    df = pd.read_csv(path)
    bcol = str(row.get("barcode_col", "barcode"))
    pcol = str(row.get("pred_label_col", "pred_label"))
    if bcol not in df.columns:
        bcol = df.columns[0]
    if pcol not in df.columns:
        candidates = [c for c in df.columns if c != bcol]
        pcol = candidates[-1]
    s = pd.Series(df[pcol].astype(str).values, index=df[bcol].astype(str))
    return s.reindex(spot_ids).fillna("MISSING").astype(str)

def build_sensor_matrix(sample_id: str, spot_ids: list[str], formal: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = formal[formal["sample_id"].astype(str).eq(sample_id)].copy()
    methods = []
    encoded_cols = []
    for _, row in rows.iterrows():
        method = str(row["method"])
        pred = load_prediction_map(row, spot_ids)
        codes, _uniques = pd.factorize(pred, sort=False)
        encoded_cols.append(codes.astype(np.int32))
        methods.append(method)
    if not encoded_cols:
        return np.zeros((len(spot_ids), 0), dtype=np.int32), []
    return np.vstack(encoded_cols).T, methods

def add_sensor_features(candidates: pd.DataFrame, canonical: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sample_id, cand in candidates.groupby("sample_id", sort=False):
        spots = canonical[canonical["sample_id"].astype(str).eq(str(sample_id))].copy()
        spots = spots[~spots["invalid_coordinate_spot"].astype(bool)].copy()
        spot_ids = spots["spot_id_normalized"].astype(str).tolist()
        spot_to_idx = {s: i for i, s in enumerate(spot_ids)}
        coords = spots[["x", "y"]].astype(float).to_numpy()
        k = min(9, len(spots))
        if len(spots) >= 2:
            nn = NearestNeighbors(n_neighbors=k).fit(coords)
            neigh = nn.kneighbors(coords, return_distance=False)
            neigh = {i: [int(x) for x in row[1:]] for i, row in enumerate(neigh)}
        else:
            neigh = {0: []}
        sensor, methods = build_sensor_matrix(str(sample_id), spot_ids, formal)
        n_methods = max(1, len(methods))
        sample_rows = []
        for row in cand.itertuples(index=False):
            cidx = [spot_to_idx[s] for s in parse_spots(row.member_spots) if s in spot_to_idx]
            cset = set(cidx)
            bidx = sorted({j for i in cidx for j in neigh.get(i, []) if j not in cset})
            if not cidx or not bidx or sensor.shape[1] == 0:
                boundary_mean = 0.0
                boundary_q75 = 0.0
                island_support = 0.0
                rare_support = 0.0
                bg_same_mean = 1.0
                candidate_label_entropy = 0.0
            else:
                method_boundary = []
                method_bg_same = []
                method_entropy = []
                method_rare = []
                for m in range(sensor.shape[1]):
                    c_labels = sensor[cidx, m]
                    b_labels = sensor[bidx, m]
                    if len(c_labels) == 0 or len(b_labels) == 0:
                        continue
                    diff = (c_labels[:, None] != b_labels[None, :]).mean()
                    method_boundary.append(float(diff))
                    method_bg_same.append(float(1.0 - diff))
                    vals, cnts = np.unique(c_labels, return_counts=True)
                    probs = cnts / max(1, cnts.sum())
                    ent = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
                    method_entropy.append(ent)
                    dominant = vals[np.argmax(cnts)]
                    cluster_size = int((sensor[:, m] == dominant).sum())
                    rare_cut = max(10, int(np.ceil(0.02 * len(spot_ids))))
                    method_rare.append(1.0 if cluster_size <= rare_cut else 0.0)
                if method_boundary:
                    mb = np.asarray(method_boundary, dtype=float)
                    boundary_mean = float(mb.mean())
                    boundary_q75 = float(np.quantile(mb, 0.75))
                    island_support = float((mb >= 0.75).sum()) / n_methods
                    bg_same_mean = float(np.mean(method_bg_same))
                    candidate_label_entropy = float(np.mean(method_entropy))
                    rare_support = float(np.mean(method_rare))
                else:
                    boundary_mean = 0.0
                    boundary_q75 = 0.0
                    island_support = 0.0
                    rare_support = 0.0
                    bg_same_mean = 1.0
                    candidate_label_entropy = 0.0
            sample_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "sensor_boundary_mean": boundary_mean,
                    "sensor_boundary_q75": boundary_q75,
                    "sensor_island_support_frac": island_support,
                    "sensor_rare_support_frac": rare_support,
                    "sensor_bg_same_mean": bg_same_mean,
                    "sensor_candidate_label_entropy": candidate_label_entropy,
                    "local_background_neighbor_count": len(bidx),
                }
            )
        feat = pd.DataFrame(sample_rows)
        frames.append(cand.merge(feat, on="candidate_id", how="left"))
        print(f"[57D_SENSOR] sample={sample_id} candidates={len(cand)} methods={len(methods)}")
    return pd.concat(frames, ignore_index=True, sort=False)
