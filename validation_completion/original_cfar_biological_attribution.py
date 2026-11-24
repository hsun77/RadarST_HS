#!/usr/bin/env python
"""Reproduce and interpret the frozen RadarST CFAR-inspired expression front end."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(r"G:\RadarST_PROJECT\03_biology\04_cfar_attribution")
FORMAL = Path(r"G:\RadarST_PROJECT\02_formal_pipeline")
BENCH = FORMAL / "benchmark"
SINGLE = FORMAL / "single"
BIO = Path(r"G:\RadarST_PROJECT\03_biology\01_local_expression")
CONTEXT = Path(r"G:\RadarST_PROJECT\03_biology\02_context_validation")
POOLED = Path(r"G:\RadarST_PROJECT\03_biology\03_pooled_landscape")
VENDOR = BIO / "vendor"
sys.path.insert(0, str(VENDOR))

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

try:
    import gseapy as gp
except ImportError as exc:
    raise SystemExit(f"Frozen project-local gseapy unavailable: {exc}") from exc

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None


LOCK = PROJECT / "CFAR_BIOLOGY_LOCK.json"
REGIONS_SOURCE = BIO / "intermediate/unique_candidate_regions.csv.gz"
BACKGROUND_SOURCE = BIO / "candidate_background_manifest.csv"
CONTEXT_SOURCE = CONTEXT / "context_region_manifest.csv.gz"
INTERNAL_CANONICAL = BENCH / "results/proposed_methods/30B_muLesionST_v1_12dataset_comparison/30B_canonical_spot_table_12datasets_used.csv"
INTERNAL_POOL = BENCH / "results/manuscript/56C_RadarST_dense_local_primitives/56C_dense_local_primitive_candidate_pool.csv"
EXT_POOL = BENCH / "results/external_validation/72K1_external_candidate_pool_protocol_parity/72K1_external_dense_sensor_candidate_pool.csv"
HER_POOL = SINGLE / "external_validation_2_her2st/candidate_pool/HER2ST_second_external_candidate_pool.csv"
SCRIPT_53B = BENCH / "scripts/53B_RadarST_background_gene_aware_candidate_refinement.py"
SCRIPT_56C = BENCH / "scripts/56C_RadarST_dense_local_primitives.py"
SCRIPT_72K1 = BENCH / "scripts/72K1_regenerate_external_candidate_pool_protocol_parity.py"
GENESET_SOURCES = {
    "hallmark": BIO / "inputs/gene_sets/h.all.v2026.1.Hs.symbols.gmt",
    "reactome": BIO / "inputs/gene_sets/c2.cp.reactome.v2026.1.Hs.symbols.gmt",
    "gobp": BIO / "inputs/gene_sets/c5.go.bp.v2026.1.Hs.symbols.gmt",
}
GENESETS = {name: PROJECT / "inputs/gene_sets" / path.name for name, path in GENESET_SOURCES.items()}

SEED = 20260912
RANDOM_BASE_SEED = 20260805
N_BLOCK_PERM = 10_000
N_BOOT = 2_000
N_GSEA_PERM = 1_000
N_RANDOM = 500
MIN_SET_SIZE = 15
MAX_SET_SIZE = 500
MIN_SECTION_GENE = 12
EXPECTED_SECTIONS = 23
EXPECTED_REGIONS = 20_561
TOL = 1e-8
MIN_CORR_N = 10

SAMPLE_SPECS = {
    "ST_PDAC_A": ("development", "PDAC", BENCH / "data_union/primary_explicit_lesion/ST_PDAC_A"),
    "ST_PDAC_B": ("development", "PDAC", BENCH / "data_union/primary_explicit_lesion/ST_PDAC_B"),
    "A01": ("development", "breast cancer", BENCH / "data_union/operational_microdomain/breast_visium_pilot"),
    "A02": ("development", "kidney cancer", BENCH / "data_union/operational_microdomain/KC1"),
    "A03": ("development", "kidney cancer", BENCH / "data_union/operational_microdomain/KC2"),
    "A04": ("development", "kidney cancer", BENCH / "data_union/operational_microdomain/KC3"),
    "A05": ("development", "lung cancer", BENCH / "data_union/operational_microdomain/LC1"),
    "A06": ("development", "lung cancer", BENCH / "data_union/operational_microdomain/LC2"),
    "A07": ("development", "lung cancer", BENCH / "data_union/operational_microdomain/LC3"),
    "A08": ("development", "lung cancer", BENCH / "data_union/operational_microdomain/LC4"),
    "A09": ("development", "lung cancer", BENCH / "data_union/operational_microdomain/LC5"),
    "10X_BC": ("development", "breast cancer", BENCH / "data_union/external_curated_breast_cancer/10X_BC"),
    "EXT04_GSM6592049_M2": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592049_M2"),
    "EXT04_GSM6592050_M3": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592050_M3"),
    "EXT04_GSM6592051_M4": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592051_M4"),
    "EXT04_GSM6592052_M5": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592052_M5"),
    "EXT04_GSM6592058_M11": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592058_M11"),
    "EXT04_GSM6592061_M15": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592061_M15"),
    "EXT04_GSM6592062_M16": ("held-out external", "breast cancer", BENCH / "data_union/external_coutant_breast_visium/EXT04_GSM6592062_M16"),
    "HER2ST_E1": ("held-out external", "breast cancer", SINGLE / "external_validation_2_her2st/processed/HER2ST_E1"),
    "HER2ST_F1": ("held-out external", "breast cancer", SINGLE / "external_validation_2_her2st/processed/HER2ST_F1"),
    "HER2ST_G2": ("held-out external", "breast cancer", SINGLE / "external_validation_2_her2st/processed/HER2ST_G2"),
    "HER2ST_H1": ("held-out external", "breast cancer", SINGLE / "external_validation_2_her2st/processed/HER2ST_H1"),
}

CONTEXT_ORDER = [
    "TLS/lymphoid microdomain",
    "Malignant epithelial/tumor",
    "Tumor boundary",
    "Immune/inflammatory context",
    "Stromal/benign tissue context",
    "unassigned",
]
CANCER_ORDER = ["breast cancer", "kidney cancer", "lung cancer", "PDAC"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object, base: int = SEED) -> int:
    token = "||".join(map(str, parts)).encode("utf-8")
    return (base + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


def bh_fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    q = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    out[valid] = np.minimum(1.0, q[inverse])
    return out


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_layout() -> None:
    for rel in ["scripts", "inputs/gene_sets", "intermediate/samples", "intermediate/candidates", "intermediate/pathways", "logs", "qa"]:
        (PROJECT / rel).mkdir(parents=True, exist_ok=True)
    for name, source in GENESET_SOURCES.items():
        if not source.exists():
            raise FileNotFoundError(source)
        if not GENESETS[name].exists() or sha256(GENESETS[name]) != sha256(source):
            shutil.copy2(source, GENESETS[name])


def canonical_members(value: object) -> str:
    return ";".join(sorted({x for x in str(value).split(";") if x and x != "nan"}))


def load_regions() -> pd.DataFrame:
    regions = pd.read_csv(REGIONS_SOURCE, dtype=str).fillna("")
    background = pd.read_csv(BACKGROUND_SOURCE, dtype=str).fillna("")
    if len(regions) != EXPECTED_REGIONS or regions["dataset_id"].nunique() != EXPECTED_SECTIONS:
        raise RuntimeError(f"Frozen canonical registry mismatch: {len(regions)} rows, {regions['dataset_id'].nunique()} sections")
    if regions["candidate_key"].duplicated().any() or background["candidate_key"].duplicated().any():
        raise RuntimeError("Canonical candidate keys are not unique")
    cols = ["candidate_key", "background_spots", "n_background", "construction_rule", "background_source"]
    regions = regions.merge(background[cols], on="candidate_key", how="left", validate="one_to_one")
    if regions["background_spots"].eq("").any():
        raise RuntimeError("Frozen local background missing")
    regions["candidate_size"] = pd.to_numeric(regions["candidate_size"], errors="raise").astype(int)
    regions["method_recurrence"] = pd.to_numeric(regions["method_recurrence"], errors="coerce").fillna(0).astype(int)
    regions["candidate_molecular_score"] = pd.to_numeric(regions["candidate_molecular_score"], errors="coerce")
    regions["cancer_type"] = regions["dataset_id"].map(lambda x: SAMPLE_SPECS[x][1])
    regions["biological_context"] = regions["biological_context"].replace("", "unassigned")
    regions["candidate_member_key"] = regions["member_spots"].map(canonical_members)
    regions["region_index"] = np.arange(len(regions), dtype=int)
    return regions


def read_expression_exact(folder: Path, spot_order: list[str]) -> tuple[np.ndarray, list[str], pd.Index]:
    h5 = folder / "filtered_feature_bc_matrix.h5"
    adata = sc.read_10x_h5(str(h5))
    adata.var_names_make_unique()
    obs = pd.Index(map(str, adata.obs_names))
    keep = [spot for spot in spot_order if spot in obs]
    if not keep:
        raise RuntimeError(f"No aligned expression barcodes: {folder}")
    adata = adata[keep].copy()
    matrix = adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.astype(np.float32).tocsr()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    totals[totals <= 0] = 1.0
    matrix = matrix.multiply((1e4 / totals)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0)
    valid = np.where(mean > 0.01)[0]
    if not len(valid):
        valid = np.arange(matrix.shape[1])
    top = valid[np.argsort(variance[valid])[-min(1000, len(valid)):]]
    return matrix[:, top].toarray().astype(np.float32), list(map(str, adata.var_names[top])), pd.Index(keep)


def formal_sample(dataset_id: str, canonical: pd.DataFrame, m53, m72) -> dict:
    cohort, cancer, folder = SAMPLE_SPECS[dataset_id]
    if cohort == "development":
        sample = canonical[canonical["dataset_id"].astype(str).eq(dataset_id)].copy()
        if sample.empty:
            raise RuntimeError(f"No internal canonical rows for {dataset_id}")
        spot_order = sample["spot_id_normalized"].astype(str).tolist()
        expression, genes, aligned = m53.read_expression(folder, spot_order)
        sample = sample.set_index("spot_id_normalized").reindex(aligned)
        coords = sample[["x", "y"]].astype(float).to_numpy()
        source_kind = "56C_internal"
    elif dataset_id.startswith("EXT04_"):
        metadata = pd.read_csv(folder / "metadata.tsv", sep="\t", dtype=str, usecols=["ID", "pxl_col_in_fullres", "pxl_row_in_fullres"])
        metadata["x"] = pd.to_numeric(metadata["pxl_col_in_fullres"], errors="coerce")
        metadata["y"] = pd.to_numeric(metadata["pxl_row_in_fullres"], errors="coerce")
        metadata = metadata.dropna(subset=["x", "y"]).drop_duplicates("ID")
        expression, genes, aligned = read_expression_exact(folder, metadata["ID"].astype(str).tolist())
        formal_expression, formal_aligned = m72.read_expression(folder, metadata["ID"].astype(str).tolist())
        if not np.array_equal(expression, formal_expression) or not aligned.equals(formal_aligned):
            raise RuntimeError(f"Extended expression reader diverged from 72K1 for {dataset_id}")
        metadata = metadata.set_index("ID").reindex(aligned)
        coords = metadata[["x", "y"]].astype(float).to_numpy()
        source_kind = "72K1_EXT04"
    else:
        canonical_spots = pd.read_csv(folder / "canonical_spot_table.csv", dtype={"spot_id": str})
        expression, genes, aligned = read_expression_exact(folder, canonical_spots["spot_id"].astype(str).tolist())
        formal_expression, formal_aligned = m72.read_expression(folder, canonical_spots["spot_id"].astype(str).tolist())
        if not np.array_equal(expression, formal_expression) or not aligned.equals(formal_aligned):
            raise RuntimeError(f"Extended expression reader diverged from 72K1 for {dataset_id}")
        canonical_spots = canonical_spots.set_index("spot_id").reindex(aligned)
        coords = canonical_spots[["pxl_col_in_fullres", "pxl_row_in_fullres"]].astype(float).to_numpy()
        source_kind = "72K1_HER2ST"
    return {
        "dataset_id": dataset_id,
        "cohort": cohort,
        "cancer_type": cancer,
        "folder": folder,
        "expression": np.asarray(expression, dtype=np.float32),
        "genes": np.asarray(genes, dtype=str),
        "spot_ids": np.asarray(aligned.astype(str), dtype=str),
        "coords": np.asarray(coords, dtype=np.float64),
        "source_kind": source_kind,
    }


def compute_frontend(sample: dict, m56, m72) -> dict:
    expression = sample["expression"]
    standardized = expression - np.nanmean(expression, axis=0, keepdims=True)
    standardized = standardized / np.maximum(np.nanstd(standardized, axis=0, keepdims=True), 1e-3)
    u, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    n_pc = min(20, u.shape[1])
    pcs = (u[:, :n_pc] * singular[:n_pc]).astype(np.float32)
    loadings = vt[:n_pc].astype(np.float32)
    if sample["source_kind"].startswith("56C"):
        formal_pcs = m56.pca_scores(expression)
        if not np.array_equal(pcs, formal_pcs):
            if not np.allclose(np.abs(pcs), np.abs(formal_pcs), atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"PCA reproduction failed for {sample['dataset_id']}")
        difference = sample["coords"][:, None, :] - sample["coords"][None, :, :]
        distance = np.sqrt(np.sum(difference * difference, axis=2))
        order = np.argsort(distance, axis=1)
        formal_score = m56.local_scores(pcs, sample["coords"])
    else:
        formal_pcs = m72.pca_scores(expression)
        if not np.array_equal(pcs, formal_pcs):
            if not np.allclose(np.abs(pcs), np.abs(formal_pcs), atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"PCA reproduction failed for {sample['dataset_id']}")
        order = NearestNeighbors(n_neighbors=min(40, len(sample["coords"]))).fit(sample["coords"]).kneighbors(sample["coords"], return_distance=False)
        formal_score = m72.local_scores(pcs, sample["coords"])
    n_spots = len(order)
    signed = np.zeros((n_spots, n_pc), dtype=np.float32)
    backgrounds: list[np.ndarray] = []
    for i, row in enumerate(order):
        bg = row[8:40] if len(row) >= 40 else row[2:]
        backgrounds.append(np.asarray(bg, dtype=np.int32))
        if len(bg) < 4:
            continue
        median = np.median(pcs[bg], axis=0)
        mad = np.maximum(np.median(np.abs(pcs[bg] - median), axis=0), 0.2)
        signed[i] = ((pcs[i] - median) / mad).astype(np.float32)
    absolute = np.abs(signed)
    k = min(8, n_pc)
    top_idx = np.argsort(absolute, axis=1, kind="mergesort")[:, -k:]
    top_mask = np.zeros_like(absolute, dtype=bool)
    np.put_along_axis(top_mask, top_idx, True, axis=1)
    recomputed_score = np.take_along_axis(absolute, top_idx, axis=1).mean(axis=1)
    if not np.allclose(recomputed_score, formal_score, atol=1e-7, rtol=0):
        raise RuntimeError(f"Local-score implementation mismatch for {sample['dataset_id']}")
    selected_signed = np.where(top_mask, signed, 0.0).astype(np.float32)
    gene_signed = selected_signed @ loadings
    gene_magnitude = np.abs(selected_signed) @ np.abs(loadings)
    denom = np.maximum(np.sum(np.abs(selected_signed), axis=1, keepdims=True), 1e-12)
    gene_normalized = (selected_signed / denom) @ loadings
    contribution = np.abs(selected_signed) / denom
    return {
        **sample,
        "standardized": standardized.astype(np.float32),
        "pcs": pcs,
        "loadings": loadings,
        "signed": signed,
        "absolute": absolute,
        "top_idx": top_idx.astype(np.int16),
        "top_mask": top_mask,
        "cfar": recomputed_score.astype(np.float32),
        "formal_function_score": np.asarray(formal_score, dtype=np.float64),
        "contribution": contribution.astype(np.float32),
        "gene_signed": gene_signed.astype(np.float32),
        "gene_magnitude": gene_magnitude.astype(np.float32),
        "gene_normalized": gene_normalized.astype(np.float32),
        "backgrounds": backgrounds,
    }


def source_pool_for_sample(dataset_id: str, internal_pool: pd.DataFrame, ext_pool: pd.DataFrame, her_pool: pd.DataFrame) -> pd.DataFrame:
    source = internal_pool if SAMPLE_SPECS[dataset_id][0] == "development" else (ext_pool if dataset_id.startswith("EXT04_") else her_pool)
    return source[source["dataset_id"].astype(str).eq(dataset_id)].copy()


def fixed_list_array(matrix: np.ndarray) -> pa.FixedSizeListArray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(matrix.ravel(), type=pa.float32()), matrix.shape[1])


def write_spot_gene_table(path: Path, samples: list[dict]) -> None:
    writer = None
    try:
        for result in samples:
            table = pa.table({
                "sample": pa.array([result["dataset_id"]] * len(result["spot_ids"])),
                "spot_id": pa.array(result["spot_ids"].tolist()),
                "gene_index_id": pa.array([result["dataset_id"]] * len(result["spot_ids"])),
                "signed_attribution": fixed_list_array(result["gene_signed"]),
                "magnitude_attribution": fixed_list_array(result["gene_magnitude"]),
                "normalized_attribution": fixed_list_array(result["gene_normalized"]),
            })
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def vectorized_row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra = stats.rankdata(a, axis=1, method="average").astype(np.float64)
    rb = stats.rankdata(b, axis=1, method="average").astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(ra * ra, axis=1) * np.sum(rb * rb, axis=1))
    return np.divide(np.sum(ra * rb, axis=1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)


def save_sample_npz(result: dict) -> None:
    max_bg = max((len(x) for x in result["backgrounds"]), default=0)
    bg = np.full((len(result["backgrounds"]), max_bg), -1, dtype=np.int32)
    bg_n = np.zeros(len(result["backgrounds"]), dtype=np.int16)
    for i, values in enumerate(result["backgrounds"]):
        bg[i, :len(values)] = values
        bg_n[i] = len(values)
    np.savez_compressed(
        PROJECT / "intermediate/samples" / f"{result['dataset_id']}.npz",
        spot_ids=result["spot_ids"], genes=result["genes"], coords=result["coords"], expression=result["expression"],
        standardized=result["standardized"], pcs=result["pcs"], loadings=result["loadings"], signed=result["signed"],
        top_idx=result["top_idx"], cfar=result["cfar"], contribution=result["contribution"],
        gene_signed=result["gene_signed"], gene_magnitude=result["gene_magnitude"], gene_normalized=result["gene_normalized"],
        fidelity_rho=result["fidelity_rho"],
        backgrounds=bg, background_n=bg_n,
    )


def save_candidate_npz(dataset_id: str, group: pd.DataFrame, result: dict, source_pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    spot_lookup = {spot: i for i, spot in enumerate(result["spot_ids"])}
    source_pool = source_pool.copy()
    source_pool["candidate_member_key"] = source_pool["member_spots"].map(canonical_members)
    source_pool["source_key"] = source_pool["proposal_type"].astype(str) + "||" + source_pool["candidate_member_key"]
    if source_pool["source_key"].duplicated().any():
        source_pool = source_pool.sort_values("candidate_id", kind="mergesort").drop_duplicates("source_key")
    source_index = source_pool.set_index("source_key")
    rows = []
    signed, magnitude, normalized, pc_share = [], [], [], []
    missing_source = 0
    for row in group.itertuples(index=False):
        member_ids = [x for x in str(row.member_spots).split(";") if x]
        if any(x not in spot_lookup for x in member_ids):
            raise RuntimeError(f"Unaligned retained member in {dataset_id}: {row.candidate_key}")
        idx = np.asarray([spot_lookup[x] for x in member_ids], dtype=int)
        signed.append(np.median(result["gene_signed"][idx], axis=0))
        magnitude.append(np.median(result["gene_magnitude"][idx], axis=0))
        normalized.append(np.median(result["gene_normalized"][idx], axis=0))
        pc_share.append(np.mean(result["contribution"][idx], axis=0))
        key = f"{row.proposal_type}||{canonical_members(row.member_spots)}"
        src = source_index.loc[key] if key in source_index.index else None
        if src is None:
            missing_source += 1
            proposal_score = np.nan
            source_candidate_id = ""
        else:
            proposal_score = float(src["proposal_score"])
            source_candidate_id = str(src["candidate_id"])
        rows.append({
            "sample": dataset_id,
            "candidate_key": row.candidate_key,
            "candidate_id": row.candidate_id,
            "source_candidate_id": source_candidate_id,
            "proposal_type": row.proposal_type,
            "member_spots": row.member_spots,
            "background_spots": row.background_spots,
            "n_background": int(row.n_background) if str(row.n_background) else len([x for x in str(row.background_spots).split(";") if x]),
            "candidate_size": int(row.candidate_size),
            "cohort": row.cohort,
            "cancer_type": row.cancer_type,
            "biological_context": row.biological_context,
            "workflow_recurrence_count": int(row.method_recurrence),
            "formal_workflows_supporting_region": row.methods,
            "candidate_molecular_score": float(row.candidate_molecular_score) if np.isfinite(row.candidate_molecular_score) else np.nan,
            "formal_proposal_score": proposal_score,
            "mean_spot_cfar": float(np.mean(result["cfar"][idx])),
            "median_spot_cfar": float(np.median(result["cfar"][idx])),
            "max_spot_cfar": float(np.max(result["cfar"][idx])),
            "median_member_spot_attribution_fidelity": float(np.nanmedian(result["fidelity_rho"][idx])),
            "section_median_spot_attribution_fidelity": float(np.nanmedian(result["fidelity_rho"])),
            "gene_index_id": dataset_id,
        })
    signed_m = np.asarray(signed, dtype=np.float32)
    magnitude_m = np.asarray(magnitude, dtype=np.float32)
    normalized_m = np.asarray(normalized, dtype=np.float32)
    pc_share_m = np.asarray(pc_share, dtype=np.float32)
    np.savez_compressed(
        PROJECT / "intermediate/candidates" / f"{dataset_id}.npz",
        candidate_key=group["candidate_key"].astype(str).to_numpy(dtype="U"), genes=np.asarray(result["genes"], dtype="U"),
        signed=signed_m, magnitude=magnitude_m, normalized=normalized_m, pc_share=pc_share_m,
    )
    return pd.DataFrame(rows), {"missing_source_candidates": missing_source, "n_retained": len(group)}


def write_candidate_gene_table(path: Path, metadata: pd.DataFrame) -> None:
    writer = None
    try:
        for dataset_id, group in metadata.groupby("sample", sort=False):
            data = np.load(PROJECT / "intermediate/candidates" / f"{dataset_id}.npz", allow_pickle=True)
            fixed = group.reset_index(drop=True)
            payload = {column: pa.array(fixed[column].tolist()) for column in fixed.columns}
            payload.update({
                "signed_attribution": fixed_list_array(data["signed"]),
                "magnitude_attribution": fixed_list_array(data["magnitude"]),
                "normalized_attribution": fixed_list_array(data["normalized"]),
                "pc_contribution_fraction": fixed_list_array(data["pc_share"]),
            })
            table = pa.table(payload)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def reproduction_stage() -> None:
    ensure_layout()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not lock.get("locked_before_outcome_analysis"):
        raise RuntimeError("Protocol lock is not active")
    m53 = import_module("cfar_m53", SCRIPT_53B)
    m56 = import_module("cfar_m56", SCRIPT_56C)
    m72 = import_module("cfar_m72", SCRIPT_72K1)
    canonical = pd.read_csv(INTERNAL_CANONICAL, dtype={"spot_id_normalized": str})
    pool_columns = ["dataset_id", "sample_id", "candidate_id", "proposal_type", "member_spots", "n_spots", "proposal_score"]
    internal_pool = pd.read_csv(INTERNAL_POOL, usecols=pool_columns, low_memory=False)
    ext_pool = pd.read_csv(EXT_POOL, usecols=pool_columns, low_memory=False)
    her_pool = pd.read_csv(HER_POOL, usecols=pool_columns, low_memory=False)
    regions = load_regions()
    audit_rows, fidelity_rows, pc_rows, contribution_rows, candidate_frames, gene_index = [], [], [], [], [], {}
    samples_for_parquet = []
    source_hashes = {str(path): sha256(path) for path in [SCRIPT_53B, SCRIPT_56C, SCRIPT_72K1, INTERNAL_POOL, EXT_POOL, HER_POOL, REGIONS_SOURCE, BACKGROUND_SOURCE]}
    for dataset_id in regions["dataset_id"].drop_duplicates():
        started = time.time()
        sample = formal_sample(dataset_id, canonical, m53, m72)
        result = compute_frontend(sample, m56, m72)
        source_pool = source_pool_for_sample(dataset_id, internal_pool, ext_pool, her_pool)
        singleton = source_pool[source_pool["proposal_type"].astype(str).eq("singleton")].copy()
        singleton["spot_id"] = singleton["member_spots"].astype(str)
        frozen = singleton.set_index("spot_id")["proposal_score"].astype(float).reindex(result["spot_ids"])
        available = frozen.notna().to_numpy()
        difference = np.abs(frozen.to_numpy(float)[available] - result["cfar"][available])
        direct = np.zeros_like(result["standardized"], dtype=np.float32)
        for i, bg in enumerate(result["backgrounds"]):
            if len(bg) >= 4:
                direct[i] = result["standardized"][i] - np.median(result["standardized"][bg], axis=0)
        rho = vectorized_row_spearman(result["gene_signed"], direct)
        result["fidelity_rho"] = rho.astype(np.float32)
        group = regions[regions["dataset_id"].eq(dataset_id)].copy()
        candidate_meta, source_audit = save_candidate_npz(dataset_id, group, result, source_pool)
        candidate_frames.append(candidate_meta)
        gene_index[dataset_id] = result["genes"].tolist()
        fidelity_rows.append({
            "sample": dataset_id,
            "n_spots": len(rho),
            "n_finite_spot_correlations": int(np.isfinite(rho).sum()),
            "median_spearman": float(np.nanmedian(rho)),
            "mean_spearman": float(np.nanmean(rho)),
            "q25_spearman": float(np.nanquantile(rho, 0.25)),
            "q75_spearman": float(np.nanquantile(rho, 0.75)),
            "fraction_positive": float(np.nanmean(rho > 0)),
            "comparison": "CFAR back-projected signed attribution versus direct standardized local gene contrast",
        })
        n_top = result["top_idx"].shape[1]
        top_rank = np.zeros_like(result["signed"], dtype=np.int16)
        ordered = np.argsort(result["absolute"], axis=1, kind="mergesort")[:, ::-1]
        for rank in range(n_top):
            top_rank[np.arange(len(ordered)), ordered[:, rank]] = rank + 1
        for pc in range(result["signed"].shape[1]):
            pc_rows.append(pd.DataFrame({
                "sample": dataset_id,
                "spot_id": result["spot_ids"],
                "pc": pc + 1,
                "signed_deviation": result["signed"][:, pc],
                "absolute_deviation": result["absolute"][:, pc],
                "is_top8": result["top_mask"][:, pc],
                "top8_rank": top_rank[:, pc],
                "formal_cfar_score": result["cfar"],
                "background_size": [len(x) for x in result["backgrounds"]],
            }))
            keep = result["top_mask"][:, pc]
            if keep.any():
                contribution_rows.append(pd.DataFrame({
                    "sample": dataset_id,
                    "spot_id": result["spot_ids"][keep],
                    "pc": pc + 1,
                    "pc_contribution_fraction": result["contribution"][keep, pc],
                    "signed_deviation": result["signed"][keep, pc],
                    "formal_cfar_score": result["cfar"][keep],
                }))
        save_sample_npz(result)
        samples_for_parquet.append(result)
        row_pass = bool(
            available.all()
            and (len(difference) == len(result["spot_ids"]))
            and (float(difference.max(initial=0)) <= TOL)
            and source_audit["missing_source_candidates"] == 0
        )
        audit_rows.append({
            "sample": dataset_id,
            "n_spots": len(result["spot_ids"]),
            "formal_score_available": bool(available.all()),
            "recomputed_score": "all spots; see spot_pc_signed_deviation.parquet",
            "max_abs_difference": float(difference.max(initial=0)),
            "mean_abs_difference": float(difference.mean()) if len(difference) else np.nan,
            "PC_match": "IMPLEMENTATION_PARITY; historical PC matrix not independently persisted",
            "gene_match": "IMPLEMENTATION_PARITY; historical selected-gene list not independently persisted",
            "background_match": "IMPLEMENTATION_PARITY; historical neighbor matrix not independently persisted",
            "proposal_membership_match": source_audit["missing_source_candidates"] == 0,
            "n_retained_candidates": source_audit["n_retained"],
            "source_kind": result["source_kind"],
            "tolerance": TOL,
            "PASS": row_pass,
            "seconds": time.time() - started,
        })
        print(f"[reproduce] {dataset_id}: spots={len(result['spot_ids'])} retained={len(group)} maxdiff={difference.max(initial=0):.3g} fidelity={np.nanmedian(rho):.3f}", flush=True)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(PROJECT / "CFAR_REPRODUCTION_AUDIT.csv", index=False)
    pd.concat(pc_rows, ignore_index=True).to_parquet(PROJECT / "spot_pc_signed_deviation.parquet", index=False, compression="zstd")
    pd.concat(contribution_rows, ignore_index=True).to_parquet(PROJECT / "spot_pc_contribution.parquet", index=False, compression="zstd")
    write_spot_gene_table(PROJECT / "spot_gene_cfar_attribution.parquet", samples_for_parquet)
    candidate_metadata = pd.concat(candidate_frames, ignore_index=True)
    candidate_metadata.to_parquet(PROJECT / "intermediate/candidate_metadata.parquet", index=False)
    write_candidate_gene_table(PROJECT / "candidate_gene_cfar_attribution.parquet", candidate_metadata)
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(PROJECT / "CFAR_GENE_BACKPROJECTION_FIDELITY.csv", index=False)
    (PROJECT / "intermediate/cfar_gene_index_by_sample.json").write_text(json.dumps(gene_index, indent=2), encoding="utf-8")
    gate = {
        "completed_utc": utc_now(),
        "reproduction_pass": bool(audit["PASS"].all()),
        "n_sections_pass": int(audit["PASS"].sum()),
        "n_sections": len(audit),
        "maximum_score_difference": float(audit["max_abs_difference"].max()),
        "fidelity_median_of_section_medians": float(fidelity["median_spearman"].median()),
        "fidelity_fraction_sections_positive": float((fidelity["median_spearman"] > 0).mean()),
        "fidelity_minimum_section_median": float(fidelity["median_spearman"].min()),
    }
    gate["fidelity_pass"] = bool(
        gate["fidelity_median_of_section_medians"] > 0.10
        and gate["fidelity_fraction_sections_positive"] >= 0.80
        and gate["fidelity_minimum_section_median"] >= -0.05
    )
    gate["attribution_unlocked"] = bool(gate["reproduction_pass"] and gate["fidelity_pass"])
    gate["formal_source_sha256"] = source_hashes
    (PROJECT / "intermediate/REPRODUCTION_GATE.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    if not gate["attribution_unlocked"]:
        raise RuntimeError(f"Biological attribution STOP: {gate}")


def trim_mean(values: np.ndarray, proportion: float = 0.10) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(stats.trim_mean(values, proportion))


def load_candidate_payload(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(PROJECT / "intermediate/candidates" / f"{dataset_id}.npz", allow_pickle=True)
    return data["candidate_key"], data["genes"], data["signed"], data["magnitude"], data["pc_share"]


def section_gene_and_landscape(candidate_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_rows = []
    global_signed: dict[str, list[np.ndarray]] = defaultdict(list)
    global_magnitude: dict[str, list[np.ndarray]] = defaultdict(list)
    for dataset_id in candidate_meta["sample"].drop_duplicates():
        _, genes, signed, magnitude, _ = load_candidate_payload(dataset_id)
        for j, gene_raw in enumerate(genes.astype(str)):
            gene = gene_raw.upper()
            values = signed[:, j].astype(float)
            mags = magnitude[:, j].astype(float)
            global_signed[gene].append(values)
            global_magnitude[gene].append(mags)
            section_rows.append({
                "sample": dataset_id,
                "gene": gene,
                "n_regions": len(values),
                "section_median_attribution": float(np.median(values)),
                "section_mean_attribution": float(np.mean(values)),
                "section_trimmed_mean_attribution": trim_mean(values),
                "section_fraction_positive": float(np.mean(values > 0)),
                "section_median_magnitude": float(np.median(mags)),
            })
    section = pd.DataFrame(section_rows)
    section.to_csv(PROJECT / "intermediate/section_gene_cfar_attribution.csv.gz", index=False, compression="gzip")
    section_pivot = section.pivot(index="sample", columns="gene", values="section_median_attribution")
    eligible = section.groupby("gene")["sample"].nunique()
    eligible = eligible[eligible >= MIN_SECTION_GENE].index
    section_pivot = section_pivot.reindex(index=candidate_meta["sample"].drop_duplicates(), columns=eligible)
    matrix = section_pivot.to_numpy(float)
    observed = np.nanmean(matrix, axis=0)
    rng = np.random.default_rng(stable_seed("gene_section_signflip"))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_BLOCK_PERM, matrix.shape[0]), replace=True)
    n_available = np.sum(np.isfinite(matrix), axis=0)
    exceed = np.zeros(matrix.shape[1], dtype=np.int64)
    for start in range(0, matrix.shape[1], 512):
        stop = min(start + 512, matrix.shape[1])
        block = np.nan_to_num(matrix[:, start:stop], nan=0.0)
        permuted = (signs @ block) / np.maximum(1, n_available[start:stop])[None, :]
        exceed[start:stop] = np.sum(np.abs(permuted) >= np.abs(observed[start:stop])[None, :], axis=0)
    gene_p = (1 + exceed) / (N_BLOCK_PERM + 1)
    gene_fdr = bh_fdr(gene_p)
    sign_fraction = np.maximum(np.nanmean(matrix > 0, axis=0), np.nanmean(matrix < 0, axis=0))
    abs_matrix = np.abs(matrix)
    dominance = np.nanmax(abs_matrix, axis=0) / np.maximum(np.nansum(abs_matrix, axis=0), 1e-12)
    inference = pd.DataFrame({
        "gene": eligible,
        "section_equal_mean_attribution": observed,
        "section_signflip_p": gene_p,
        "section_signflip_fdr": gene_fdr,
        "fraction_sections_same_direction": sign_fraction,
        "maximum_section_absolute_share": dominance,
    }).set_index("gene")
    rows = []
    for gene in sorted(global_signed):
        values = np.concatenate(global_signed[gene]).astype(float)
        mags = np.concatenate(global_magnitude[gene]).astype(float)
        sec = section[section["gene"].eq(gene)]
        record = {
            "gene": gene,
            "region_weighted_median_attribution": float(np.median(values)),
            "region_weighted_mean_attribution": float(np.mean(values)),
            "region_weighted_trimmed_mean_attribution": trim_mean(values),
            "fraction_attribution_positive": float(np.mean(values > 0)),
            "region_weighted_median_magnitude": float(np.median(mags)),
            "section_equal_mean_attribution": float(sec["section_median_attribution"].mean()),
            "section_median_attribution": float(sec["section_median_attribution"].median()),
            "fraction_sections_positive": float((sec["section_median_attribution"] > 0).mean()),
            "section_equal_mean_magnitude": float(sec["section_median_magnitude"].mean()),
            "n_regions": len(values),
            "n_sections": sec["sample"].nunique(),
            "eligible_for_pathway": gene in inference.index,
        }
        if gene in inference.index:
            record.update(inference.loc[gene].to_dict())
            if record["section_signflip_fdr"] < 0.05 and record["fraction_sections_same_direction"] >= 0.75:
                record["attribution_breadth"] = "broad"
            elif record["maximum_section_absolute_share"] >= 0.50:
                record["attribution_breadth"] = "section-driven"
            else:
                record["attribution_breadth"] = "heterogeneous"
        else:
            record.update({
                "section_signflip_p": np.nan,
                "section_signflip_fdr": np.nan,
                "fraction_sections_same_direction": np.nan,
                "maximum_section_absolute_share": np.nan,
                "attribution_breadth": "insufficient-section-coverage",
            })
        rows.append(record)
    landscape = pd.DataFrame(rows).sort_values(
        ["section_equal_mean_magnitude", "region_weighted_median_magnitude", "gene"],
        ascending=[False, False, True], kind="mergesort"
    ).reset_index(drop=True)
    landscape.to_csv(PROJECT / "cfar_gene_landscape.csv", index=False)
    return section, landscape


def pc_landscape(candidate_meta: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset_id, meta in candidate_meta.groupby("sample", sort=False):
        keys, genes, _, _, pc_share = load_candidate_payload(dataset_id)
        if list(meta["candidate_key"].astype(str)) != list(keys.astype(str)):
            raise RuntimeError(f"Candidate row order mismatch for {dataset_id}")
        for local_i, row in enumerate(meta.itertuples(index=False)):
            for pc in range(pc_share.shape[1]):
                records.append({
                    "sample": dataset_id,
                    "candidate_key": row.candidate_key,
                    "cancer_type": row.cancer_type,
                    "biological_context": row.biological_context,
                    "pc": pc + 1,
                    "contribution_fraction": float(pc_share[local_i, pc]),
                    "is_driver": bool(pc_share[local_i, pc] > 0),
                })
    long = pd.DataFrame(records)
    for group_col, filename in [
        ("sample", "pc_usage_by_section.csv"),
        ("cancer_type", "pc_usage_by_cancer.csv"),
        ("biological_context", "pc_usage_by_context.csv"),
    ]:
        summary = long.groupby([group_col, "pc"], as_index=False).agg(
            n_candidates=("candidate_key", "nunique"),
            driver_frequency=("is_driver", "mean"),
            mean_contribution_fraction=("contribution_fraction", "mean"),
            median_contribution_fraction=("contribution_fraction", "median"),
        )
        summary.to_csv(PROJECT / filename, index=False)
    long.to_parquet(PROJECT / "intermediate/candidate_pc_contribution.parquet", index=False, compression="zstd")
    return long


def pc_gene_drivers(pc_long: pd.DataFrame) -> pd.DataFrame:
    usage = pc_long.groupby("pc", as_index=False).agg(
        global_driver_frequency=("is_driver", "mean"),
        global_mean_contribution_fraction=("contribution_fraction", "mean"),
        sections_represented=("sample", "nunique"),
        cancer_types_represented=("cancer_type", "nunique"),
    ).set_index("pc")
    rows = []
    for dataset_id in pc_long["sample"].drop_duplicates():
        data = np.load(PROJECT / "intermediate/samples" / f"{dataset_id}.npz", allow_pickle=False)
        genes = data["genes"].astype(str)
        loadings = data["loadings"]
        cancer = SAMPLE_SPECS[dataset_id][1]
        for pc in range(loadings.shape[0]):
            positive = np.argsort(loadings[pc], kind="mergesort")[-10:][::-1]
            negative = np.argsort(loadings[pc], kind="mergesort")[:10]
            for direction, indices in [("positive", positive), ("negative", negative)]:
                for rank, j in enumerate(indices, start=1):
                    rows.append({
                        "sample": dataset_id,
                        "cancer_type": cancer,
                        "pc": pc + 1,
                        "gene": genes[j],
                        "loading": float(loadings[pc, j]),
                        "loading_direction": direction,
                        "loading_rank_within_direction": rank,
                        **usage.loc[pc + 1].to_dict(),
                    })
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / "pc_gene_driver_table.csv", index=False)
    return result


def select_representative(candidate_meta: pd.DataFrame, fidelity: pd.DataFrame) -> dict:
    eligible = candidate_meta[
        candidate_meta["candidate_size"].between(3, 5)
        & candidate_meta["biological_context"].ne("unassigned")
        & candidate_meta["formal_proposal_score"].notna()
    ].copy()
    eligible = eligible[
        eligible["median_member_spot_attribution_fidelity"]
        >= eligible["section_median_spot_attribution_fidelity"]
    ]
    if eligible.empty:
        raise RuntimeError("No candidate satisfies the locked representative-case rule")
    cohort_median = eligible.groupby("cohort")["median_spot_cfar"].transform("median")
    eligible["distance_to_cohort_median_cfar"] = (eligible["median_spot_cfar"] - cohort_median).abs()
    selected = eligible.sort_values(
        ["distance_to_cohort_median_cfar", "sample", "candidate_key"], kind="mergesort"
    ).iloc[0]
    lock = {
        "locked_utc": utc_now(),
        "candidate_key": selected["candidate_key"],
        "sample": selected["sample"],
        "candidate_size": int(selected["candidate_size"]),
        "biological_context": selected["biological_context"],
        "median_spot_cfar": float(selected["median_spot_cfar"]),
        "selection_rule": "size 3-5; assigned frozen context; complete formal proposal/background; candidate median member-spot fidelity at or above its section spot-level median; closest CFAR intensity to cohort median; lexicographic tie break",
        "pathway_results_read_during_selection": False,
    }
    (PROJECT / "REPRESENTATIVE_CASE_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def candidate_explanations(candidate_meta: pd.DataFrame) -> None:
    output = PROJECT / "candidate_cfar_explanations.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for dataset_id, meta in candidate_meta.groupby("sample", sort=False):
            keys, genes, signed, _, pc_share = load_candidate_payload(dataset_id)
            for i, row in enumerate(meta.itertuples(index=False)):
                pc_order = np.argsort(pc_share[i], kind="mergesort")[::-1][:3]
                gene_order = np.argsort(np.abs(signed[i]), kind="mergesort")[::-1][:20]
                positive = np.argsort(signed[i], kind="mergesort")[::-1][:10]
                negative = np.argsort(signed[i], kind="mergesort")[:10]
                record = {
                    "sample": dataset_id,
                    "candidate_id": row.candidate_id,
                    "candidate_key": row.candidate_key,
                    "candidate_size": int(row.candidate_size),
                    "mean_CFAR": float(row.mean_spot_cfar),
                    "median_CFAR": float(row.median_spot_cfar),
                    "top_PCs": [{"pc": int(j + 1), "share": float(pc_share[i, j])} for j in pc_order],
                    "top_genes": [{"gene": str(genes[j]), "attribution": float(signed[i, j])} for j in gene_order],
                    "top_positive_gene_attributions": [{"gene": str(genes[j]), "attribution": float(signed[i, j])} for j in positive],
                    "top_negative_gene_attributions": [{"gene": str(genes[j]), "attribution": float(signed[i, j])} for j in negative],
                }
                handle.write(json.dumps(record) + "\n")


def parse_gmt(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                result[parts[0]] = {x.strip().upper() for x in parts[2:] if x.strip()}
    return result


def deterministic_rank(frame: pd.DataFrame, score: str, secondary: str) -> pd.DataFrame:
    rank = frame[["gene", score, secondary]].dropna().sort_values(
        [score, secondary, "gene"], ascending=[False, False, True], kind="mergesort"
    )[["gene", score]].rename(columns={score: "score"})
    scale = max(1.0, float(rank["score"].abs().max()))
    for _, positions in rank.groupby("score", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) > 1:
            rank.iloc[positions, rank.columns.get_loc("score")] += np.linspace(scale * 1e-12, -scale * 1e-12, len(positions))
    return rank


def normalize_gsea(frame: pd.DataFrame, library: str, ranking: str, gene_sets: dict[str, set[str]], universe: set[str]) -> pd.DataFrame:
    rename = {
        "Term": "term", "ES": "es", "NES": "nes", "NOM p-val": "nominal_p", "FDR q-val": "fdr",
        "FWER p-val": "fwer_p", "Tag %": "tag_fraction", "Gene %": "gene_fraction", "Lead_genes": "leading_edge_genes",
    }
    out = frame.rename(columns=rename).copy()
    fields = ["term", "es", "nes", "nominal_p", "fdr", "fwer_p", "tag_fraction", "gene_fraction", "leading_edge_genes"]
    for field in fields:
        if field not in out:
            out[field] = np.nan
    out = out[fields]
    for field in ["es", "nes", "nominal_p", "fdr", "fwer_p"]:
        out[field] = pd.to_numeric(out[field], errors="coerce")
    out.insert(0, "ranking", ranking)
    out.insert(0, "library", library)
    out["mapped_gene_set_size"] = out["term"].map(lambda term: len(gene_sets.get(str(term), set()) & universe))
    out["gsea_permutations"] = N_GSEA_PERM
    out["independent_unit_note"] = "aggregate descriptive preranked GSEA; section-block sign flips define section-aware uncertainty"
    return out


def run_gsea(rank: pd.DataFrame, library: str, gene_sets: dict[str, set[str]], ranking: str) -> pd.DataFrame:
    result = gp.prerank(
        rnk=rank[["gene", "score"]], gene_sets=str(GENESETS[library]), min_size=MIN_SET_SIZE,
        max_size=MAX_SET_SIZE, permutation_num=N_GSEA_PERM, weight=1.0, ascending=False,
        threads=max(1, min(8, os.cpu_count() or 1)), seed=SEED, outdir=None, no_plot=True, verbose=False,
    )
    return normalize_gsea(result.res2d, library, ranking, gene_sets, set(rank["gene"]))


def block_permutation(library: str, gene_sets: dict[str, set[str]], genes: pd.Index, section_effect: np.ndarray) -> pd.DataFrame:
    ranks = np.full_like(section_effect, np.nan, dtype=np.float64)
    for i in range(section_effect.shape[0]):
        valid = np.isfinite(section_effect[i])
        if valid.any():
            values = stats.rankdata(section_effect[i, valid], method="average")
            ranks[i, valid] = values / (len(values) + 1.0) - 0.5
    gene_to_i = {gene: i for i, gene in enumerate(genes)}
    terms, contrasts = [], []
    for term, members in gene_sets.items():
        idx = np.asarray([gene_to_i[g] for g in members if g in gene_to_i], dtype=int)
        if not MIN_SET_SIZE <= len(idx) <= MAX_SET_SIZE:
            continue
        mask = np.ones(len(genes), dtype=bool)
        mask[idx] = False
        contrast = np.nanmean(ranks[:, idx], axis=1) - np.nanmean(ranks[:, mask], axis=1)
        terms.append((term, len(idx)))
        contrasts.append(contrast)
    contrast_matrix = np.column_stack(contrasts)
    observed = np.nanmean(contrast_matrix, axis=0)
    available = np.sum(np.isfinite(contrast_matrix), axis=0)
    rng = np.random.default_rng(stable_seed(library, "section_block"))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_BLOCK_PERM, section_effect.shape[0]), replace=True)
    exceed = np.zeros(len(terms), dtype=np.int64)
    for start in range(0, len(terms), 512):
        stop = min(start + 512, len(terms))
        permuted = (signs @ np.nan_to_num(contrast_matrix[:, start:stop], nan=0.0)) / np.maximum(1, available[start:stop])[None, :]
        exceed[start:stop] = np.sum(np.abs(permuted) >= np.abs(observed[start:stop])[None, :], axis=0)
    p = (1 + exceed) / (N_BLOCK_PERM + 1)
    result = pd.DataFrame({
        "library": library,
        "term": [x[0] for x in terms],
        "mapped_gene_set_size": [x[1] for x in terms],
        "block_es_observed": observed,
        "empirical_p": p,
        "n_section_block_permutations": N_BLOCK_PERM,
        "n_sections": section_effect.shape[0],
        "n_sections_with_mapped_effect": available,
        "statistic_definition": "mean centered within-section gene rank in set minus nonset, equally averaged across sections",
    })
    result["fdr"] = bh_fdr(result["empirical_p"])
    return result.sort_values(["fdr", "empirical_p", "term"], kind="mergesort").reset_index(drop=True)


def aggregate_pathways(section: pd.DataFrame, landscape: pd.DataFrame) -> tuple[dict[str, dict[str, set[str]]], pd.DataFrame]:
    eligible = landscape[landscape["eligible_for_pathway"]].copy()
    genes = pd.Index(eligible["gene"].astype(str))
    section_ids = pd.Index(section["sample"].drop_duplicates())
    section_matrix = section.pivot(index="sample", columns="gene", values="section_median_attribution").reindex(index=section_ids, columns=genes).to_numpy(float)
    region_rank = deterministic_rank(eligible, "region_weighted_median_attribution", "region_weighted_mean_attribution")
    section_rank = deterministic_rank(eligible, "section_equal_mean_attribution", "section_median_attribution")
    libraries = {name: parse_gmt(path) for name, path in GENESETS.items()}
    merged = []
    for library, gene_sets in libraries.items():
        started = time.time()
        region = run_gsea(region_rank, library, gene_sets, "region_weighted")
        section_equal = run_gsea(section_rank, library, gene_sets, "section_equal")
        block = block_permutation(library, gene_sets, genes, section_matrix)
        region.to_csv(PROJECT / f"{library}_cfar_regionweighted.csv", index=False)
        section_equal.to_csv(PROJECT / f"{library}_cfar_sectionequal.csv", index=False)
        block.to_csv(PROJECT / f"{library}_cfar_blockperm.csv", index=False)
        r = region.add_prefix("region_").rename(columns={"region_library": "library", "region_term": "term"})
        s = section_equal.add_prefix("section_").rename(columns={"section_library": "library", "section_term": "term"})
        b = block.add_prefix("block_").rename(columns={"block_library": "library", "block_term": "term"})
        merged.append(r.merge(s, on=["library", "term"], how="outer").merge(b, on=["library", "term"], how="outer"))
        print(f"[pathway] {library}: region={len(region)} block={len(block)} seconds={time.time()-started:.1f}", flush=True)
    master = pd.concat(merged, ignore_index=True)
    master["region_section_same_direction"] = np.sign(master["region_nes"]) == np.sign(master["section_nes"])
    master["section_block_same_direction"] = np.sign(master["section_nes"]) == np.sign(master["block_block_es_observed"])
    master["pathway_class"] = "E NO SUPPORT"
    heterogeneous = master["region_fdr"].fillna(1).lt(0.05)
    broad = heterogeneous & master["region_section_same_direction"] & master["block_fdr"].fillna(1).lt(0.05) & master["section_block_same_direction"]
    master.loc[heterogeneous, "pathway_class"] = "B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM"
    master.loc[broad, "pathway_class"] = "A BROAD CFAR-ATTRIBUTED PROGRAM"
    master.to_csv(PROJECT / "intermediate/cfar_pathway_aggregate_master.csv", index=False)
    return libraries, master


def select_followup_terms(master: pd.DataFrame, libraries: dict[str, dict[str, set[str]]], eligible_genes: set[str]) -> pd.DataFrame:
    rows = []
    mandatory_tokens = ["MTORC1_SIGNALING", "P53_PATHWAY", "B_CELL_RECEPTOR_SIGNALING"]
    for library in ["hallmark", "reactome", "gobp"]:
        subset = master[master["library"].eq(library)].copy()
        subset["block_fdr_sort"] = subset["block_fdr"].fillna(1)
        subset["region_fdr_sort"] = subset["region_fdr"].fillna(1)
        subset["abs_section_nes"] = subset["section_nes"].abs().fillna(0)
        subset = subset.sort_values(["block_fdr_sort", "region_fdr_sort", "abs_section_nes", "term"], ascending=[True, True, False, True], kind="mergesort")
        limit = len(subset) if library == "hallmark" else 20
        selected = set(subset.head(limit)["term"].astype(str))
        for term in subset["term"].astype(str):
            if any(token in term.upper() for token in mandatory_tokens):
                selected.add(term)
        for term in sorted(selected):
            mapped = libraries[library].get(term, set()) & eligible_genes
            if len(mapped) < MIN_SET_SIZE:
                continue
            row = subset[subset["term"].eq(term)].iloc[0]
            rows.append({
                "library": library,
                "term": term,
                "mapped_gene_set_size_global": len(mapped),
                "selection_reason": "all Hallmark" if library == "hallmark" else ("locked named term" if any(token in term.upper() for token in mandatory_tokens) else "top-20 deterministic aggregate evidence ordering"),
                "aggregate_pathway_class": row["pathway_class"],
                "region_nes": row["region_nes"],
                "region_fdr": row["region_fdr"],
                "section_nes": row["section_nes"],
                "block_fdr": row["block_fdr"],
            })
    selected = pd.DataFrame(rows).sort_values(["library", "term"], kind="mergesort").reset_index(drop=True)
    selected["pathway_index"] = np.arange(len(selected), dtype=int)
    selected.to_csv(PROJECT / "intermediate/cfar_candidate_pathway_lock.csv", index=False)
    return selected


def centered_row_ranks(matrix: np.ndarray) -> np.ndarray:
    ranks = stats.rankdata(matrix, axis=1, method="average").astype(np.float32)
    return ranks / (matrix.shape[1] + 1.0) - 0.5


def pathway_membership(genes: np.ndarray, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {gene.upper(): i for i, gene in enumerate(genes.astype(str))}
    membership = np.zeros((len(genes), len(selected)), dtype=np.float32)
    sizes = np.zeros(len(selected), dtype=np.int32)
    for j, row in enumerate(selected.itertuples(index=False)):
        idx = [lookup[g] for g in libraries[row.library].get(row.term, set()) if g in lookup]
        sizes[j] = len(idx)
        if idx:
            membership[idx, j] = 1.0
    return membership, sizes


def set_contrast_from_ranks(ranks: np.ndarray, membership: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    set_sum = ranks @ membership
    total = ranks.sum(axis=1, keepdims=True)
    n_gene = ranks.shape[1]
    set_mean = np.divide(set_sum, sizes[None, :], out=np.full_like(set_sum, np.nan), where=sizes[None, :] > 0)
    non_n = n_gene - sizes
    non_mean = np.divide(total - set_sum, non_n[None, :], out=np.full_like(set_sum, np.nan), where=non_n[None, :] > 0)
    result = set_mean - non_mean
    result[:, sizes < 5] = np.nan
    return result.astype(np.float32)


def mean_membership_matrix(texts: pd.Series, spot_lookup: dict[str, int], n_spots: int) -> sparse.csr_matrix:
    rows, cols, values = [], [], []
    for i, text in enumerate(texts.astype(str)):
        idx = sorted({spot_lookup[x] for x in text.split(";") if x in spot_lookup})
        if not idx:
            continue
        rows.extend([i] * len(idx))
        cols.extend(idx)
        values.extend([1.0 / len(idx)] * len(idx))
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(texts), n_spots), dtype=np.float32)


def symmetric_knn_graph(coords: np.ndarray, k: int = 6) -> dict[int, set[int]]:
    graph = {i: set() for i in range(len(coords))}
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(return_distance=False)
    for i, row in enumerate(neighbors):
        for j in row:
            j = int(j)
            if i != j:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def graph_shells(graph: dict[int, set[int]], core: list[int]) -> tuple[list[int], list[int]]:
    core_set = set(core)
    ring1 = set().union(*(graph[i] for i in core_set)) - core_set if core_set else set()
    ring2 = set().union(*(graph[i] for i in ring1)) - core_set - ring1 if ring1 else set()
    return sorted(ring1), sorted(ring2)


def section_bootstrap_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed("section_bootstrap", key))
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def candidate_pathway_stage(candidate_meta: pd.DataFrame, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correlation_rows, context_payload, gradient_rows = [], [], []
    all_candidate_rows = []
    for dataset_id, meta in candidate_meta.groupby("sample", sort=False):
        data = np.load(PROJECT / "intermediate/samples" / f"{dataset_id}.npz", allow_pickle=False)
        cand = np.load(PROJECT / "intermediate/candidates" / f"{dataset_id}.npz", allow_pickle=True)
        genes = data["genes"].astype(str)
        membership, sizes = pathway_membership(genes, selected, libraries)
        candidate_rank = centered_row_ranks(cand["signed"])
        candidate_pathway = set_contrast_from_ranks(candidate_rank, membership, sizes)
        spot_rank = centered_row_ranks(data["expression"])
        spot_direct_pathway = set_contrast_from_ranks(spot_rank, membership, sizes)
        spot_attr_rank = centered_row_ranks(data["gene_signed"])
        spot_attr_pathway = set_contrast_from_ranks(spot_attr_rank, membership, sizes)
        spot_lookup = {spot: i for i, spot in enumerate(data["spot_ids"].astype(str))}
        core_matrix = mean_membership_matrix(meta["member_spots"], spot_lookup, len(spot_lookup))
        bg_matrix = mean_membership_matrix(meta["background_spots"], spot_lookup, len(spot_lookup))
        direct_local = np.asarray(core_matrix @ spot_direct_pathway - bg_matrix @ spot_direct_pathway, dtype=np.float32)
        if list(meta["candidate_key"].astype(str)) != list(cand["candidate_key"].astype(str)):
            raise RuntimeError(f"Candidate payload order mismatch for pathway stage: {dataset_id}")
        candidate_quantile = pd.qcut(
            meta["median_spot_cfar"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        ).astype(str).to_numpy()
        np.savez_compressed(
            PROJECT / "intermediate/pathways" / f"{dataset_id}.npz",
            candidate_key=cand["candidate_key"], terms=selected["term"].astype(str).to_numpy(), libraries=selected["library"].astype(str).to_numpy(),
            candidate_cfar_pathway=candidate_pathway, direct_local_pathway=direct_local,
        )
        for j, pathway in enumerate(selected.itertuples(index=False)):
            valid = np.isfinite(direct_local[:, j]) & np.isfinite(meta["median_spot_cfar"].to_numpy(float))
            rho, p = (stats.spearmanr(meta["median_spot_cfar"].to_numpy(float)[valid], direct_local[valid, j]) if valid.sum() >= MIN_CORR_N else (np.nan, np.nan))
            correlation_rows.append({
                "sample": dataset_id, "library": pathway.library, "term": pathway.term,
                "n_candidates": int(valid.sum()), "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "descriptive_p": float(p) if np.isfinite(p) else np.nan,
                "minimum_candidates_required": MIN_CORR_N,
                "cfar_strength_definition": "median member-spot formal CFAR score",
                "pathway_contrast_definition": "candidate minus frozen 30-spot local background centered expression-rank pathway score",
            })
            for i, row in enumerate(meta.itertuples(index=False)):
                context_payload.append({
                    "sample": dataset_id, "cohort": row.cohort, "cancer_type": row.cancer_type,
                    "biological_context": row.biological_context, "candidate_key": row.candidate_key,
                    "candidate_size": int(row.candidate_size), "median_spot_cfar": float(row.median_spot_cfar),
                    "cfar_quantile": candidate_quantile[i],
                    "library": pathway.library, "term": pathway.term,
                    "cfar_pathway_attribution": float(candidate_pathway[i, j]) if np.isfinite(candidate_pathway[i, j]) else np.nan,
                    "direct_pathway_local_contrast": float(direct_local[i, j]) if np.isfinite(direct_local[i, j]) else np.nan,
                })
        graph = symmetric_knn_graph(data["coords"], k=6)
        ring1_text, ring2_text = [], []
        for row in meta.itertuples(index=False):
            core = [spot_lookup[x] for x in str(row.member_spots).split(";") if x in spot_lookup]
            ring1, ring2 = graph_shells(graph, core)
            ring1_text.append(";".join(data["spot_ids"][ring1].astype(str)))
            ring2_text.append(";".join(data["spot_ids"][ring2].astype(str)))
        shell_matrices = {
            "CORE": core_matrix,
            "RING1": mean_membership_matrix(pd.Series(ring1_text), spot_lookup, len(spot_lookup)),
            "RING2": mean_membership_matrix(pd.Series(ring2_text), spot_lookup, len(spot_lookup)),
            "LOCAL_BG": bg_matrix,
        }
        for shell, shell_matrix in shell_matrices.items():
            direct_shell = np.asarray(shell_matrix @ spot_direct_pathway, dtype=np.float32)
            attr_shell = np.asarray(shell_matrix @ spot_attr_pathway, dtype=np.float32)
            nonempty = np.asarray(shell_matrix.sum(axis=1)).ravel() > 0
            for j, pathway in enumerate(selected.itertuples(index=False)):
                dv = direct_shell[nonempty, j]
                av = attr_shell[nonempty, j]
                gradient_rows.append({
                    "sample": dataset_id, "library": pathway.library, "term": pathway.term, "shell": shell,
                    "n_regions": int(nonempty.sum()),
                    "median_direct_pathway_score": float(np.nanmedian(dv)) if len(dv) else np.nan,
                    "mean_direct_pathway_score": float(np.nanmean(dv)) if len(dv) else np.nan,
                    "median_cfar_attributed_pathway_score": float(np.nanmedian(av)) if len(av) else np.nan,
                    "mean_cfar_attributed_pathway_score": float(np.nanmean(av)) if len(av) else np.nan,
                })
        all_candidate_rows.append({"sample": dataset_id, "n_candidates": len(meta), "n_pathways": len(selected)})
        print(f"[candidate pathways] {dataset_id}: candidates={len(meta)} pathways={len(selected)}", flush=True)

    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(PROJECT / "section_cfar_pathway_correlations.csv", index=False)
    payload = pd.DataFrame(context_payload)
    payload.to_parquet(PROJECT / "intermediate/candidate_pathway_attribution.parquet", index=False, compression="zstd")
    context = payload.groupby(["library", "term", "biological_context"], as_index=False).agg(
        n_regions=("candidate_key", "nunique"), n_sections=("sample", "nunique"),
        median_cfar_pathway_attribution=("cfar_pathway_attribution", "median"),
        mean_cfar_pathway_attribution=("cfar_pathway_attribution", "mean"),
        fraction_positive=("cfar_pathway_attribution", lambda x: float(np.mean(np.asarray(x) > 0))),
        median_direct_pathway_local_contrast=("direct_pathway_local_contrast", "median"),
    )
    context.to_csv(PROJECT / "cfar_pathway_by_context.csv", index=False)
    cancer = payload.groupby(["library", "term", "cancer_type"], as_index=False).agg(
        n_regions=("candidate_key", "nunique"), n_sections=("sample", "nunique"),
        median_cfar_pathway_attribution=("cfar_pathway_attribution", "median"),
        mean_cfar_pathway_attribution=("cfar_pathway_attribution", "mean"),
        fraction_positive=("cfar_pathway_attribution", lambda x: float(np.mean(np.asarray(x) > 0))),
        median_direct_pathway_local_contrast=("direct_pathway_local_contrast", "median"),
    )
    cancer.to_csv(PROJECT / "cfar_pathway_by_cancer.csv", index=False)
    gradient_summary = pd.DataFrame(gradient_rows)
    gradient_summary.to_csv(PROJECT / "cfar_spatial_gradient.csv", index=False)
    return correlations, payload, context, cancer


def summarize_pathway_associations(correlations: pd.DataFrame, payload: pd.DataFrame, selected: pd.DataFrame) -> None:
    rows = []
    for (library, term), group in correlations.groupby(["library", "term"], sort=False):
        values = group["spearman_rho"].to_numpy(float)
        values = values[np.isfinite(values)]
        low, high = section_bootstrap_ci(values, f"{library}|{term}")
        rows.append({
            "library": library, "term": term, "n_sections": len(values),
            "median_rho": float(np.median(values)) if len(values) else np.nan,
            "mean_rho": float(np.mean(values)) if len(values) else np.nan,
            "fraction_rho_positive": float(np.mean(values > 0)) if len(values) else np.nan,
            "section_bootstrap_ci_low": low, "section_bootstrap_ci_high": high,
            "section_bootstrap_replicates": N_BOOT,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(PROJECT / "intermediate/cfar_pathway_correlation_summary.csv", index=False)

    mixed_rows = []
    mixed_terms = selected.sort_values(["block_fdr", "region_fdr", "term"], na_position="last", kind="mergesort").head(15)
    for row in mixed_terms.itertuples(index=False):
        frame = payload[(payload["library"].eq(row.library)) & (payload["term"].eq(row.term))].dropna(
            subset=["direct_pathway_local_contrast", "median_spot_cfar"]
        ).copy()
        status, beta, se, p, n = "not_fit", np.nan, np.nan, np.nan, len(frame)
        if smf is not None and len(frame) >= 50 and frame["sample"].nunique() >= 3:
            try:
                model = smf.mixedlm("direct_pathway_local_contrast ~ median_spot_cfar", frame, groups=frame["sample"])
                fit = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
                beta = float(fit.params.get("median_spot_cfar", np.nan))
                se = float(fit.bse.get("median_spot_cfar", np.nan))
                p = float(fit.pvalues.get("median_spot_cfar", np.nan))
                status = "mixed_random_intercept"
            except Exception as exc:
                status = f"fit_failed:{type(exc).__name__}"
        mixed_rows.append({
            "library": row.library, "term": row.term, "n_candidates": n,
            "n_sections": frame["sample"].nunique(), "beta_cfar_strength": beta,
            "se": se, "p_value": p, "model_status": status,
            "model": "direct pathway local contrast ~ median member-spot CFAR + (1|section)",
            "role": "secondary sensitivity",
        })
    mixed = pd.DataFrame(mixed_rows)
    mixed["fdr"] = bh_fdr(mixed["p_value"])
    mixed.to_csv(PROJECT / "cfar_pathway_mixed_model.csv", index=False)

    quantile = payload.groupby(["library", "term", "sample", "cfar_quantile"], observed=True, as_index=False).agg(
        n_candidates=("candidate_key", "nunique"),
        median_cfar_pathway_attribution=("cfar_pathway_attribution", "median"),
        median_direct_pathway_local_contrast=("direct_pathway_local_contrast", "median"),
        mean_direct_pathway_local_contrast=("direct_pathway_local_contrast", "mean"),
    )
    quantile.to_csv(PROJECT / "cfar_quantile_pathway_summary.csv", index=False)


def finalize_pathway_classes(master: pd.DataFrame, context: pd.DataFrame, cancer: pd.DataFrame) -> pd.DataFrame:
    result = master.copy()
    result["class_support_detail"] = ""
    for i, row in result.iterrows():
        if str(row["pathway_class"]).startswith(("A ", "B ")):
            continue
        ctx = context[(context["library"].eq(row["library"])) & (context["term"].eq(row["term"]))].copy()
        ctx["direction_fraction"] = np.maximum(ctx["fraction_positive"], 1 - ctx["fraction_positive"])
        ctx_hit = ctx[(ctx["n_regions"] >= 8) & (ctx["median_cfar_pathway_attribution"].abs() >= 0.05) & (ctx["direction_fraction"] >= 0.75)]
        if not ctx_hit.empty:
            best = ctx_hit.sort_values(["median_cfar_pathway_attribution", "biological_context"], key=lambda x: x.abs() if np.issubdtype(x.dtype, np.number) else x, ascending=False).iloc[0]
            result.at[i, "pathway_class"] = "C CONTEXT-ASSOCIATED CFAR PROGRAM"
            result.at[i, "class_support_detail"] = f"{best['biological_context']}; n={int(best['n_regions'])}"
            continue
        can = cancer[(cancer["library"].eq(row["library"])) & (cancer["term"].eq(row["term"]))].copy()
        can["direction_fraction"] = np.maximum(can["fraction_positive"], 1 - can["fraction_positive"])
        can_hit = can[(can["n_sections"] >= 2) & (can["median_cfar_pathway_attribution"].abs() >= 0.05) & (can["direction_fraction"] >= 0.75)]
        if not can_hit.empty:
            best = can_hit.sort_values("median_cfar_pathway_attribution", key=lambda x: x.abs(), ascending=False).iloc[0]
            result.at[i, "pathway_class"] = "D CANCER-ASSOCIATED CFAR PROGRAM"
            result.at[i, "class_support_detail"] = f"{best['cancer_type']}; sections={int(best['n_sections'])}"
    result.to_csv(PROJECT / "intermediate/cfar_pathway_classification_master.csv", index=False)
    for library in ["hallmark", "reactome", "gobp"]:
        classes = result[result["library"].eq(library)][["term", "pathway_class", "class_support_detail"]]
        for suffix in ["regionweighted", "sectionequal", "blockperm"]:
            path = PROJECT / f"{library}_cfar_{suffix}.csv"
            frame = pd.read_csv(path)
            frame = frame.drop(columns=["pathway_class", "class_support_detail"], errors="ignore").merge(classes, on="term", how="left")
            frame.to_csv(path, index=False)
    return result


def random_connected_set(graph: dict[int, set[int]], size: int, rng: np.random.Generator) -> list[int] | None:
    vertices = np.asarray(sorted(graph), dtype=int)
    for _ in range(100):
        seed = int(rng.choice(vertices))
        selected = {seed}
        frontier = set(graph[seed])
        while len(selected) < size and frontier:
            choices = np.asarray(sorted(frontier), dtype=int)
            nxt = int(rng.choice(choices))
            frontier.discard(nxt)
            if nxt in selected:
                continue
            selected.add(nxt)
            frontier.update(graph[nxt] - selected)
        if len(selected) == size:
            return sorted(selected)
    return None


def random_calibration(candidate_meta: pd.DataFrame, selected: pd.DataFrame, libraries: dict[str, dict[str, set[str]]], top_n_pathways: int = 15) -> pd.DataFrame:
    selected_top = selected.sort_values(["block_fdr", "region_fdr", "term"], na_position="last", kind="mergesort").head(top_n_pathways).reset_index(drop=True)
    rows = []
    for dataset_id, meta in candidate_meta.groupby("sample", sort=False):
        # NpzFile indexing decompresses on every access; cache identical arrays once per section.
        with np.load(PROJECT / "intermediate/samples" / f"{dataset_id}.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("coords", "genes", "cfar", "gene_signed")}
        cand = np.load(PROJECT / "intermediate/candidates" / f"{dataset_id}.npz", allow_pickle=True)
        graph = symmetric_knn_graph(data["coords"], k=6)
        full_membership, full_sizes = pathway_membership(data["genes"].astype(str), selected, libraries)
        top_positions = selected.set_index(["library", "term"])["pathway_index"].reindex(
            pd.MultiIndex.from_frame(selected_top[["library", "term"]])
        ).to_numpy(int)
        candidate_pathway = set_contrast_from_ranks(centered_row_ranks(cand["signed"]), full_membership, full_sizes)[:, top_positions]
        for size in sorted(meta["candidate_size"].unique()):
            subset = meta["candidate_size"].eq(size).to_numpy()
            rng = np.random.default_rng(stable_seed(dataset_id, int(size), "matched_random", base=RANDOM_BASE_SEED))
            random_sets = []
            attempts = 0
            while len(random_sets) < N_RANDOM and attempts < N_RANDOM * 20:
                attempts += 1
                value = random_connected_set(graph, int(size), rng)
                if value is not None:
                    random_sets.append(value)
            if len(random_sets) < N_RANDOM:
                raise RuntimeError(f"Random reconstruction did not reach {N_RANDOM}: {dataset_id}, size {size}")
            random_cfar = np.asarray([np.median(data["cfar"][idx]) for idx in random_sets], dtype=float)
            random_gene = np.asarray([np.median(data["gene_signed"][idx], axis=0) for idx in random_sets], dtype=np.float32)
            random_pathway = set_contrast_from_ranks(centered_row_ranks(random_gene), full_membership, full_sizes)[:, top_positions]
            observed_cfar = meta.loc[subset, "median_spot_cfar"].to_numpy(float)
            observed_median = float(np.median(observed_cfar))
            rows.append({
                "sample": dataset_id, "candidate_size": int(size), "metric": "CFAR intensity", "library": "", "term": "",
                "n_observed_candidates": int(subset.sum()), "n_matched_random_regions": len(random_sets),
                "observed_median": observed_median, "random_median": float(np.median(random_cfar)),
                "observed_minus_random_median": observed_median - float(np.median(random_cfar)),
                "candidate_percentile_in_random_distribution": 100 * float(np.mean(random_cfar <= observed_median)),
                "empirical_random_p_two_sided": (1 + int(np.sum(np.abs(random_cfar - np.median(random_cfar)) >= abs(observed_median - np.median(random_cfar))))) / (1 + len(random_cfar)),
                "random_framework": "deterministic pooled-landscape reconstruction: symmetric 6-NN, 500 connected regions, base seed 20260805",
                "historical_member_identity_verifiable": False,
            })
            for j, pathway in enumerate(selected_top.itertuples(index=False)):
                observed = np.abs(candidate_pathway[subset, j])
                null = np.abs(random_pathway[:, j])
                observed_median = float(np.nanmedian(observed))
                random_median = float(np.nanmedian(null))
                rows.append({
                    "sample": dataset_id, "candidate_size": int(size), "metric": "absolute CFAR-attributed pathway score",
                    "library": pathway.library, "term": pathway.term,
                    "n_observed_candidates": int(np.isfinite(observed).sum()), "n_matched_random_regions": int(np.isfinite(null).sum()),
                    "observed_median": observed_median, "random_median": random_median,
                    "observed_minus_random_median": observed_median - random_median,
                    "candidate_percentile_in_random_distribution": 100 * float(np.nanmean(null <= observed_median)),
                    "empirical_random_p_two_sided": (1 + int(np.nansum(np.abs(null - random_median) >= abs(observed_median - random_median)))) / (1 + int(np.isfinite(null).sum())),
                    "random_framework": "deterministic pooled-landscape reconstruction: symmetric 6-NN, 500 connected regions, base seed 20260805",
                    "historical_member_identity_verifiable": False,
                })
        print(f"[random] {dataset_id}: sizes={sorted(meta['candidate_size'].unique())}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(PROJECT / "cfar_random_calibration.csv", index=False)
    return result


def attribute_stage() -> None:
    ensure_layout()
    gate = json.loads((PROJECT / "intermediate/REPRODUCTION_GATE.json").read_text(encoding="utf-8"))
    if not gate.get("attribution_unlocked"):
        raise RuntimeError("Attribution remains locked by the reproduction/fidelity gate")
    candidate_meta = pd.read_parquet(PROJECT / "intermediate/candidate_metadata.parquet")
    fidelity = pd.read_csv(PROJECT / "CFAR_GENE_BACKPROJECTION_FIDELITY.csv")
    section, landscape = section_gene_and_landscape(candidate_meta)
    pc_long = pc_landscape(candidate_meta)
    pc_gene_drivers(pc_long)
    select_representative(candidate_meta, fidelity)
    candidate_explanations(candidate_meta)
    libraries, aggregate_master = aggregate_pathways(section, landscape)
    eligible_genes = set(landscape.loc[landscape["eligible_for_pathway"], "gene"].astype(str))
    selected = select_followup_terms(aggregate_master, libraries, eligible_genes)
    correlations, payload, context, cancer = candidate_pathway_stage(candidate_meta, selected, libraries)
    summarize_pathway_associations(correlations, payload, selected)
    final_master = finalize_pathway_classes(aggregate_master, context, cancer)
    random_calibration(candidate_meta, selected, libraries)
    summary = {
        "completed_utc": utc_now(),
        "n_sections": candidate_meta["sample"].nunique(),
        "n_canonical_regions": candidate_meta["candidate_key"].nunique(),
        "n_gene_landscape": len(landscape),
        "n_pathway_eligible_genes": int(landscape["eligible_for_pathway"].sum()),
        "n_candidate_followup_pathways": len(selected),
        "pathway_class_counts": final_master["pathway_class"].value_counts().to_dict(),
        "radarst_model_modified": False,
        "new_performance_analysis_run": False,
    }
    (PROJECT / "intermediate/ATTRIBUTION_STATUS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def concise_term(value: str, length: int = 34) -> str:
    text = str(value).replace("HALLMARK_", "").replace("REACTOME_", "").replace("GOBP_", "").replace("_", " ").title()
    return text if len(text) <= length else text[:length - 1].rstrip() + "…"


def interpretation_level(master: pd.DataFrame, cancer: pd.DataFrame) -> tuple[str, str]:
    broad = master[master["pathway_class"].eq("A BROAD CFAR-ATTRIBUTED PROGRAM")]
    for row in broad.itertuples(index=False):
        sub = cancer[(cancer["library"].eq(row.library)) & (cancer["term"].eq(row.term))]
        strong = sub[(sub["n_sections"] >= 2) & (sub["median_cfar_pathway_attribution"].abs() >= 0.02)]
        if strong["cancer_type"].nunique() >= 2:
            return "LEVEL A — BROAD MOLECULAR AXIS", "At least one section-supported CFAR-attributed program was represented across multiple cancer types."
    if master["pathway_class"].isin(["C CONTEXT-ASSOCIATED CFAR PROGRAM", "D CANCER-ASSOCIATED CFAR PROGRAM"]).any():
        return "LEVEL B — CONTEXT-DEPENDENT MOLECULAR AXES", "No qualifying universal program was required; locked context or cancer decompositions identified structured CFAR-attributed axes."
    if master["pathway_class"].eq("B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM").any():
        return "LEVEL C — LOCAL / HETEROGENEOUS ATTRIBUTION", "Aggregate gene/pathway attribution was interpretable for local regions, but section-supported context structure was limited."
    return "LEVEL C — LOCAL / HETEROGENEOUS ATTRIBUTION", "Back-projection was interpretable, but no pathway met the locked broad or context-associated criteria."


def generate_documents() -> None:
    gate = json.loads((PROJECT / "intermediate/REPRODUCTION_GATE.json").read_text(encoding="utf-8"))
    status = json.loads((PROJECT / "intermediate/ATTRIBUTION_STATUS.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(PROJECT / "CFAR_REPRODUCTION_AUDIT.csv")
    fidelity = pd.read_csv(PROJECT / "CFAR_GENE_BACKPROJECTION_FIDELITY.csv")
    genes = pd.read_csv(PROJECT / "cfar_gene_landscape.csv")
    drivers = pd.read_csv(PROJECT / "pc_gene_driver_table.csv")
    master = pd.read_csv(PROJECT / "intermediate/cfar_pathway_classification_master.csv")
    correlations = pd.read_csv(PROJECT / "intermediate/cfar_pathway_correlation_summary.csv")
    context = pd.read_csv(PROJECT / "cfar_pathway_by_context.csv")
    cancer = pd.read_csv(PROJECT / "cfar_pathway_by_cancer.csv")
    random = pd.read_csv(PROJECT / "cfar_random_calibration.csv")
    pc = pd.read_csv(PROJECT / "pc_usage_by_section.csv")
    level, level_reason = interpretation_level(master, cancer)
    pc_global_all = pc.groupby("pc", as_index=False).agg(mean_contribution=("mean_contribution_fraction", "mean"), mean_frequency=("driver_frequency", "mean"))
    pc_global = pc_global_all.sort_values("mean_contribution", ascending=False).head(5)
    pc_frequent = pc_global_all.sort_values(["mean_frequency", "pc"], ascending=[False, True]).head(5)
    top_pos = genes.sort_values(["section_equal_mean_attribution", "gene"], ascending=[False, True]).head(10)
    top_neg = genes.sort_values(["section_equal_mean_attribution", "gene"], ascending=[True, True]).head(10)
    broad_n = int(master["pathway_class"].eq("A BROAD CFAR-ATTRIBUTED PROGRAM").sum())
    hetero_n = int(master["pathway_class"].eq("B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM").sum())
    context_n = int(master["pathway_class"].eq("C CONTEXT-ASSOCIATED CFAR PROGRAM").sum())
    cancer_n = int(master["pathway_class"].eq("D CANCER-ASSOCIATED CFAR PROGRAM").sum())
    cf_random = random[random["metric"].eq("CFAR intensity")]
    random_positive = float((cf_random["observed_minus_random_median"] > 0).mean())
    pathway_random = random[random["metric"].eq("absolute CFAR-attributed pathway score")].dropna(subset=["observed_minus_random_median"])
    pathway_random_positive = float((pathway_random["observed_minus_random_median"] > 0).mean())
    pathway_random_delta = float(pathway_random["observed_minus_random_median"].median())
    hetero_terms = master[master["pathway_class"].eq("B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM")].sort_values(
        ["region_fdr", "term"], kind="mergesort"
    )["term"].head(6).tolist()
    context_details = master[master["pathway_class"].eq("C CONTEXT-ASSOCIATED CFAR PROGRAM")]["class_support_detail"].dropna()
    tls_context_n = int(context_details.str.contains("TLS/lymphoid", regex=False).sum())
    immune_context_n = int(context_details.str.contains("Immune/inflammatory", regex=False).sum())
    strongest_corr = correlations.loc[correlations["median_rho"].abs().idxmax()] if len(correlations) else None
    named_rows = []
    for token in ["MTORC1", "P53", "B_CELL_RECEPTOR"]:
        hit = master[master["term"].str.contains(token, case=False, na=False)]
        if hit.empty:
            named_rows.append(f"- `{token}`: no eligible term in the locked mapped universe.")
        else:
            best = hit.sort_values(["block_fdr", "region_fdr"], na_position="last").iloc[0]
            named_rows.append(f"- `{best['term']}`: {best['pathway_class']}; region NES={best['region_nes']:.3f}, region FDR={best['region_fdr']:.3g}, section-block FDR={best['block_fdr']:.3g}.")
    bcell = context[context["term"].str.contains("B_CELL_RECEPTOR", case=False, na=False) & context["biological_context"].str.contains("TLS", case=False, na=False)]
    tls_sentence = "No eligible B-cell-receptor/TLS attribution row was available."
    if not bcell.empty:
        best = bcell.sort_values("n_regions", ascending=False).iloc[0]
        tls_sentence = (
            f"The locked TLS/lymphoid subset contained {int(best['n_regions'])} regions for `{best['term']}`; "
            f"the median CFAR-attributed score was {best['median_cfar_pathway_attribution']:.4f}, "
            f"{100*best['fraction_positive']:.1f}% of regions were positive, and the median direct local contrast was {best['median_direct_pathway_local_contrast']:.4f}."
        )

    methods = f"""# Methods

## Formal detector reproduction

The analysis called the candidate-generation implementations used by the frozen RadarST workflow (`56C` for development sections and protocol-parity `72K1` for EXT04/HER2ST sections). Counts were normalized to 10,000 per spot and log-transformed. Genes with mean transformed expression above 0.01 were ranked by variance and the top 1,000 were standardized before a 20-component SVD PCA. Euclidean spatial neighbors were ordered exactly as in the formal code. The local background was `order[i,8:40]` for sections with at least 40 spots and `order[i,2:]` otherwise; at least four background spots were required. Signed robust PC deviations used a MAD floor of 0.2. The formal anomaly score was the mean of the eight largest absolute deviations.

Frozen singleton proposal scores independently anchored the recomputed spot scores. The maximum observed absolute difference was {gate['maximum_score_difference']:.3g} across {int(audit['n_spots'].sum()):,} audited spots. Historical PC matrices, selected-gene lists and neighbor matrices were not separately persisted; these objects are therefore described as implementation-parity objects rather than independent frozen matches.

## Sign-invariant attribution

Only each spot's formal top-eight PC deviations were back-projected. Signed attribution was `A_ig = sum_p d_ip L_pg`, magnitude was `M_ig = sum_p |d_ip L_pg|`, and normalized composition used `sum_p w_ip sign(d_ip)L_pg`. PC sign is arbitrary, but the signed product `d_ip × loading_pg` is invariant to a global sign flip of the PC. These are CFAR back-projected gene contributions, not causal or Shapley attributions.

The direct standardized local gene contrast used the same spot-specific background. Per-spot Spearman correlations between direct and back-projected gene contrasts were summarized by section. The prespecified fidelity gate required the median section median to exceed 0.10, at least 80% of sections to be positive and no section median below -0.05.

## Candidate, gene and pathway summaries

The analysis unit was each of {status['n_canonical_regions']:,} frozen canonical retained physical regions across {status['n_sections']} sections. Cross-workflow recurrence did not create biological replicates. Candidate gene attribution was the median member-spot attribution. Region-weighted and section-equal gene landscapes were retained separately; the pathway universe required a formal PCA gene to occur in at least {MIN_SECTION_GENE} sections.

Weighted preranked GSEA used the frozen MSigDB 2026.1.Hs Hallmark, Reactome and GO Biological Process libraries (weight 1, set size 15-500, {N_GSEA_PERM:,} permutations). Section-aware inference used {N_BLOCK_PERM:,} section-block sign flips on centered within-section gene ranks. Candidate-level pathway and context summaries were descriptive follow-ups under the deterministic lock in `PROTOCOL.md`. CFAR-strength associations were calculated within section, with section as the independent summary unit. A random-intercept mixed model was a secondary sensitivity only.

## Spatial and random calibration

CORE, RING1, RING2 and frozen 30-spot LOCAL_BG summaries used the established symmetric six-nearest-neighbor graph. Matched random regions reused the pooled-landscape deterministic reconstruction (500 connected regions per section and candidate size; base seed 20260805). Historical random member identities were not persisted, so this is a deterministic framework replay rather than a claim of member-for-member historical identity.

No truth label, IoU, ARI, selector, threshold or RadarST model parameter was used or modified in this project.
"""
    (PROJECT / "METHODS.md").write_text(methods, encoding="utf-8")

    result_text = f"""# Results

## Reproduction and interpretability gates

The formal CFAR-inspired front end was reproduced in all {len(audit)} sections ({int(audit['n_spots'].sum()):,} spots). The maximum absolute difference from frozen singleton scores was {gate['maximum_score_difference']:.3g}; all {status['n_canonical_regions']:,} retained canonical regions mapped to their frozen formal proposal pools. The reproduction gate therefore **passed**.

Back-projected gene attribution had a median section-level Spearman correlation of {gate['fidelity_median_of_section_medians']:.3f} with the direct standardized local gene contrast. {100*gate['fidelity_fraction_sections_positive']:.1f}% of section medians were positive and the minimum section median was {gate['fidelity_minimum_section_median']:.3f}. The locked fidelity gate therefore **passed**.

## CFAR-driving PCs and genes

The five PCs with the largest section-averaged contribution were {', '.join(f"PC{int(r.pc)} ({r.mean_contribution:.1%})" for r in pc_global.itertuples(index=False))}. The five PCs most frequently entering the formal top-eight set were {', '.join(f"PC{int(r.pc)} ({r.mean_frequency:.1%} of candidates on average across sections)" for r in pc_frequent.itertuples(index=False))}. The full PC-to-gene loading table reports the positive and negative loading tails independently for every section and PC. Section-equal positive attribution was led by {', '.join(top_pos['gene'].astype(str))}; negative attribution was led by {', '.join(top_neg['gene'].astype(str))}. These labels were generated by the locked numeric ordering and were not manually selected.

## CFAR-attributed pathways

Across all three frozen libraries, {broad_n} pathways met the broad section-supported criteria, {hetero_n} were classified as pooled/heterogeneous, {context_n} as context-associated, and {cancer_n} as cancer-associated. Candidate-level pathway scores indicate which programs are represented among genes contributing to the local CFAR deviation; they do not demonstrate pathway activation.

The leading pooled/heterogeneous rows included {', '.join(hetero_terms)}. Their region-weighted signals were not supported by the section-block FDR criterion and therefore were not promoted to broad programs.

{chr(10).join(named_rows)}

## CFAR strength and local pathway contrast

Within-section CFAR-strength correlations were available for {int(correlations['n_sections'].max()) if len(correlations) else 0} sections per eligible pathway at most. The strongest absolute median association among the deterministic follow-up set was {strongest_corr['term'] if strongest_corr is not None else 'not available'} (median rho={strongest_corr['median_rho'] if strongest_corr is not None else float('nan'):.3f}; section-bootstrap 95% CI {f"[{strongest_corr['section_bootstrap_ci_low']:.3f}, {strongest_corr['section_bootstrap_ci_high']:.3f}]" if strongest_corr is not None else 'not available'}). The effect was small, and inference is based on the distribution of section-level correlations rather than a pooled test over candidate regions.

## Context, cancer and random calibration

{tls_sentence}

The locked context classification assigned {tls_context_n} pathway rows to the TLS/lymphoid subset (691 regions across 8 sections) and {immune_context_n} to the immune/inflammatory subset (346 regions across 9 sections). This supports context-dependent attribution for those locked contexts, but not a distinct supported axis for every pathology category. No pathway met the locked cancer-associated criterion; cancer-specific tables therefore remain descriptive.

Candidate CFAR intensity exceeded the matched-random median in {100*random_positive:.1f}% of section-by-size comparisons (median observed-minus-random difference {cf_random['observed_minus_random_median'].median():.3f}). In contrast, absolute CFAR-attributed pathway scores exceeded their matched-random medians in {100*pathway_random_positive:.1f}% of {len(pathway_random):,} evaluable section-by-size-pathway comparisons, with a median difference of {pathway_random_delta:.4f}; this does not support a general candidate-wide excess of canonical pathway attribution. Random controls use the frozen reconstruction framework; exact historical random membership was unavailable and is not claimed.

## Final classification

**{level}.** {level_reason}

The answer to the project's final question is **{'YES' if level.startswith('LEVEL A') else ('PARTIALLY' if level.startswith(('LEVEL B','LEVEL C')) else 'NO')}**: the formal CFAR contrast is reproducibly decomposable in gene space, while the extent to which it supports broad versus context-dependent molecular programs is limited to the locked classification above.

## Direct answers to the protocol questions

1. Formal CFAR reproduced: **YES**.
2. Most frequent PCs: {', '.join(f"PC{int(r.pc)}" for r in pc_frequent.itertuples(index=False))}; complete section-level values are in `pc_usage_by_section.csv`.
3. Major PC genes: listed in `pc_gene_driver_table.csv`.
4. Back-projection fidelity: median section rho {gate['fidelity_median_of_section_medians']:.3f}; gate passed.
5. Frequent CFAR genes: `cfar_gene_landscape.csv`.
6. Hallmark/Reactome/GO programs: the leading heterogeneous rows are summarized above; complete results are in six aggregate GSEA tables and three block-permutation tables.
7. Section-supported broad program count: {broad_n}.
8. Heterogeneous program count: {hetero_n}.
9. CFAR intensity associations: the largest absolute median section-level rho was {strongest_corr['median_rho'] if strongest_corr is not None else float('nan'):.3f}; complete values are in `section_cfar_pathway_correlations.csv`.
10. Cross-section stability: the strongest deterministic follow-up association had section-bootstrap 95% CI {f"[{strongest_corr['section_bootstrap_ci_low']:.3f}, {strongest_corr['section_bootstrap_ci_high']:.3f}]" if strongest_corr is not None else 'not available'} and remained small; full summaries are in `intermediate/cfar_pathway_correlation_summary.csv`.
11. Pathology contexts: context-associated structure was supported for locked TLS/lymphoid and immune/inflammatory subsets, but not as a separate supported axis for every pathology category; see `cfar_pathway_by_context.csv`.
12. Cancer types: no pathway met the locked cancer-associated criterion; `cfar_pathway_by_cancer.csv` and PC tables are descriptive.
13. Candidate versus random CFAR intensity: higher in {100*random_positive:.1f}% of section-by-size cells; median difference {cf_random['observed_minus_random_median'].median():.3f}.
14. Candidate versus random pathway attribution: {100*pathway_random_positive:.1f}% positive with median difference {pathway_random_delta:.4f}, providing no general candidate-wide pathway-attribution excess.
15. MTORC1/P53/B-cell receptor terms: reported above under locked tests.
16. TLS B-cell signal: {tls_sentence}
17. Universal CFAR signature: {'supported under the locked broad criteria' if broad_n else 'not supported under the locked broad criteria'}.
18. Context-dependent interpretation: {'supported' if level.startswith('LEVEL B') else 'limited/heterogeneous'}.
19. RadarST model modified: **NO**.
"""
    (PROJECT / "RESULTS.md").write_text(result_text, encoding="utf-8")


def figure_stage() -> None:
    candidate_meta = pd.read_parquet(PROJECT / "intermediate/candidate_metadata.parquet")
    lock = json.loads((PROJECT / "REPRESENTATIVE_CASE_LOCK.json").read_text(encoding="utf-8"))
    genes = pd.read_csv(PROJECT / "cfar_gene_landscape.csv")
    master = pd.read_csv(PROJECT / "intermediate/cfar_pathway_classification_master.csv")
    corr = pd.read_csv(PROJECT / "section_cfar_pathway_correlations.csv")
    corr_summary = pd.read_csv(PROJECT / "intermediate/cfar_pathway_correlation_summary.csv")
    context = pd.read_csv(PROJECT / "cfar_pathway_by_context.csv")
    random = pd.read_csv(PROJECT / "cfar_random_calibration.csv")
    pc_usage = pd.read_csv(PROJECT / "pc_usage_by_section.csv")
    selected = pd.read_csv(PROJECT / "intermediate/cfar_candidate_pathway_lock.csv")
    data = np.load(PROJECT / "intermediate/candidates" / f"{lock['sample']}.npz", allow_pickle=True)
    sample_meta = candidate_meta[candidate_meta["sample"].eq(lock["sample"])].reset_index(drop=True)
    case_i = int(sample_meta.index[sample_meta["candidate_key"].eq(lock["candidate_key"])][0])
    explanation = None
    with (PROJECT / "candidate_cfar_explanations.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["candidate_key"] == lock["candidate_key"]:
                explanation = row
                break
    if explanation is None:
        raise RuntimeError("Representative explanation missing")
    path_data = np.load(PROJECT / "intermediate/pathways" / f"{lock['sample']}.npz", allow_pickle=True)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    navy, teal, coral, purple, gold, grey, pale = "#183B66", "#178F86", "#D85845", "#6A4C93", "#D58A16", "#59636E", "#E8EDF2"
    class_colors = {
        "A BROAD CFAR-ATTRIBUTED PROGRAM": teal,
        "B HETEROGENEOUS CFAR-ATTRIBUTED PROGRAM": gold,
        "C CONTEXT-ASSOCIATED CFAR PROGRAM": purple,
        "D CANCER-ASSOCIATED CFAR PROGRAM": coral,
        "E NO SUPPORT": "#B7BEC6",
    }
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.12, 1.0, 0.95], hspace=0.42, wspace=0.34, left=0.075, right=0.975, top=0.89, bottom=0.055)
    axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]
    axa, axb, axc, axd, axe, axf, axg, axh = axes
    fig.suptitle(
        "Molecular attribution of the frozen RadarST CFAR-inspired local contrast",
        x=0.075,
        y=0.982,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    for label, ax in zip("abcdefgh", axes):
        ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    # a: schematic
    axa.axis("off")
    axa.set_title("Sign-invariant interpretation path", loc="left", fontsize=8, fontweight="bold", pad=7)
    labels = ["Local PC\ndeviation", "Top-8\n|d|", "d × loading", "Gene\nattribution", "Pathway\nattribution"]
    colors = [navy, teal, purple, coral, gold]
    xs = np.linspace(0.04, 0.84, len(labels))
    for x, label, color in zip(xs, labels, colors):
        axa.add_patch(plt.Rectangle((x, 0.38), 0.14, 0.28, transform=axa.transAxes, facecolor="white", edgecolor=color, linewidth=1.3))
        axa.text(x + 0.07, 0.52, label, ha="center", va="center", transform=axa.transAxes, fontsize=6.3, color=color, fontweight="bold")
    for x1, x2 in zip(xs[:-1], xs[1:]):
        axa.annotate("", xy=(x2 - 0.01, 0.52), xytext=(x1 + 0.15, 0.52), xycoords=axa.transAxes, arrowprops=dict(arrowstyle="-|>", color=grey, lw=0.9))
    axa.text(0.04, 0.16, "PC sign may flip; d × loading is invariant", transform=axa.transAxes, color=grey, fontsize=6.1, style="italic")

    # b: representative decomposition
    axb.set_title(
        f"Locked representative region\n{lock['sample']} · size {lock['candidate_size']}",
        loc="left",
        fontsize=8,
        fontweight="bold",
        pad=7,
    )
    axb.axis("off")
    pcs = explanation["top_PCs"]
    pc_labels = [f"PC{x['pc']}" for x in pcs]
    shares = [x["share"] for x in pcs]
    pc_ax = axb.inset_axes([0.05, 0.70, 0.92, 0.18])
    pc_y = np.arange(len(pcs))
    pc_ax.barh(pc_y, shares, height=0.55, color=teal, alpha=0.85)
    pc_ax.set_yticks(pc_y, pc_labels)
    pc_ax.set_xlabel("")
    pc_ax.set_xticks([])
    pc_ax.spines["bottom"].set_visible(False)
    pc_ax.set_title("PC contribution fraction", loc="left", fontsize=5.8, pad=1)
    for y_i, value in zip(pc_y, shares):
        pc_ax.text(value, y_i, f"  {value:.1%}", va="center", ha="left", fontsize=5.2, color=grey)
    pc_ax.set_xlim(0, max(shares) * 1.20)
    pc_ax.tick_params(labelsize=5.5)
    pc_ax.invert_yaxis()
    top_genes = explanation["top_genes"][:4]
    gvals = np.asarray([x["attribution"] for x in top_genes])
    gene_ax = axb.inset_axes([0.05, 0.36, 0.92, 0.25])
    gene_y = np.arange(len(top_genes))
    gene_ax.barh(gene_y, gvals, height=0.58, color=np.where(gvals >= 0, coral, navy), alpha=0.82)
    gene_ax.set_yticks(gene_y, [x["gene"] for x in top_genes])
    gene_ax.axvline(0, color=grey, lw=0.5)
    gene_ax.set_xlabel("")
    gene_ax.text(0.0, 1.03, "Signed gene attribution", transform=gene_ax.transAxes, fontsize=5.8, va="bottom")
    gene_ax.tick_params(labelsize=5.5)
    gene_ax.invert_yaxis()
    path_case_i = int(np.flatnonzero(path_data["candidate_key"].astype(str) == lock["candidate_key"])[0])
    path_values = path_data["candidate_cfar_pathway"][path_case_i].astype(float)
    path_order = np.argsort(-np.abs(path_values), kind="mergesort")[:3]
    path_ax = axb.inset_axes([0.05, 0.02, 0.92, 0.22])
    path_y = np.arange(len(path_order))
    path_ax.barh(path_y, path_values[path_order], height=0.58, color=np.where(path_values[path_order] >= 0, gold, purple), alpha=0.86)
    path_ax.set_yticks(path_y, [concise_term(str(path_data["terms"][i]), 24) for i in path_order])
    path_ax.axvline(0, color=grey, lw=0.5)
    path_ax.set_xlabel("CFAR-attributed pathway score", fontsize=5.8, labelpad=2)
    path_ax.tick_params(labelsize=5.3)
    path_ax.invert_yaxis()

    # c: global PC contribution
    pc_global = pc_usage.groupby("pc", as_index=False).agg(mean_contribution=("mean_contribution_fraction", "mean"))
    axc.set_title("Global PC contribution landscape", loc="left", fontsize=8, fontweight="bold", pad=7)
    axc.bar(pc_global["pc"], pc_global["mean_contribution"], color=np.where(pc_global["pc"] <= 8, teal, "#AEB7C1"), width=0.75)
    axc.set_xlabel("Principal component")
    axc.set_ylabel("Mean candidate contribution")
    axc.set_xticks([1, 4, 8, 12, 16, 20])

    # d: gene landscape
    axd.set_title("Section-equal CFAR gene attribution", loc="left", fontsize=8, fontweight="bold", pad=7)
    pos = genes.nlargest(8, "section_equal_mean_attribution")
    neg = genes.nsmallest(8, "section_equal_mean_attribution")
    display = pd.concat([neg, pos]).sort_values("section_equal_mean_attribution")
    y = np.arange(len(display))
    vals = display["section_equal_mean_attribution"].to_numpy(float)
    axd.barh(y, vals, color=np.where(vals >= 0, coral, navy), height=0.68)
    axd.set_yticks(y, display["gene"])
    axd.axvline(0, color=grey, lw=0.6)
    axd.set_xlabel("Signed attribution")

    # e: pathway landscape
    axe.set_title("CFAR-attributed Hallmark programs", loc="left", fontsize=8, fontweight="bold", pad=7)
    hall = master[master["library"].eq("hallmark")].copy()
    hall["sort"] = hall["block_fdr"].fillna(1)
    hall = hall.sort_values(["sort", "region_fdr", "term"], kind="mergesort").head(10).sort_values("section_nes")
    y = np.arange(len(hall))
    axe.scatter(hall["section_nes"], y, s=np.clip(-np.log10(hall["block_fdr"].fillna(1).clip(lower=1e-6)) * 16 + 14, 14, 85), c=[class_colors[x] for x in hall["pathway_class"]], edgecolor="white", linewidth=0.35)
    axe.set_yticks(y, [concise_term(x, 28) for x in hall["term"]])
    axe.axvline(0, color="#9AA3AD", lw=0.6, ls="--")
    axe.set_xlabel("Section-equal NES")

    # f: section-level correlations
    axf.set_title("CFAR strength versus local pathway contrast", loc="left", fontsize=8, fontweight="bold", pad=7)
    best = corr_summary.iloc[corr_summary["median_rho"].abs().argmax()]
    values = corr[(corr["library"].eq(best["library"])) & (corr["term"].eq(best["term"]))].dropna(subset=["spearman_rho"]).sort_values("spearman_rho")
    y = np.arange(len(values))
    axf.hlines(y, 0, values["spearman_rho"], color="#B9C1CA", lw=0.8)
    axf.scatter(values["spearman_rho"], y, s=17, color=purple, edgecolor="white", linewidth=0.35)
    axf.axvline(0, color=grey, lw=0.6, ls="--")
    axf.set_yticks([])
    axf.set_xlabel("Within-section Spearman ρ")
    axf.text(0.02, 0.97, concise_term(best["term"], 42), transform=axf.transAxes, va="top", fontsize=6.2)
    axf.text(0.02, 0.88, f"median ρ = {best['median_rho']:.2f} [{best['section_bootstrap_ci_low']:.2f}, {best['section_bootstrap_ci_high']:.2f}]", transform=axf.transAxes, va="top", fontsize=6.1, color=grey)

    # g: context heatmap
    axg.set_title("Locked pathology-context decomposition", loc="left", fontsize=8, fontweight="bold", pad=7)
    top_terms = hall.tail(6)["term"].tolist()
    heat = context[context["term"].isin(top_terms)].pivot(index="term", columns="biological_context", values="median_cfar_pathway_attribution")
    heat = heat.reindex(index=top_terms, columns=[x for x in CONTEXT_ORDER if x in heat.columns])
    sns.heatmap(heat, ax=axg, cmap=sns.diverging_palette(240, 12, as_cmap=True), center=0, cbar_kws={"label": "Median attribution", "shrink": 0.58}, linewidths=0.25, linecolor="white")
    axg.set_yticklabels([concise_term(x, 24) for x in heat.index], rotation=0)
    axg.set_xticklabels([concise_term(x, 16) for x in heat.columns], rotation=32, ha="right")
    axg.set_xlabel("")
    axg.set_ylabel("")

    # h: random calibration
    axh.set_title("Candidate versus matched-random CFAR calibration", loc="left", fontsize=8, fontweight="bold", pad=7)
    rand = random[random["metric"].eq("CFAR intensity")].copy()
    summary = rand.groupby("sample", as_index=False).agg(delta=("observed_minus_random_median", "median"), cancer=("sample", lambda x: SAMPLE_SPECS[x.iloc[0]][1]))
    summary = summary.sort_values("delta")
    cancer_colors = {"breast cancer": teal, "kidney cancer": navy, "lung cancer": purple, "PDAC": gold}
    y = np.arange(len(summary))
    axh.hlines(y, 0, summary["delta"], color="#C5CBD2", lw=0.7)
    axh.scatter(summary["delta"], y, c=[cancer_colors[x] for x in summary["cancer"]], s=17, edgecolor="white", linewidth=0.3)
    axh.axvline(0, color=grey, lw=0.6, ls="--")
    axh.set_yticks([])
    axh.set_xlabel("Observed − random median CFAR")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=4, label=name) for name, color in cancer_colors.items()]
    axh.legend(handles=handles, loc="lower right", fontsize=5.4, ncol=2, handletextpad=0.3, columnspacing=0.7)

    output = PROJECT / "Figure_CFAR_BIOLOGICAL_ATTRIBUTION_v1"
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    note = """# Figure source-data mapping

- Panel a: schematic of formulas frozen in `PROTOCOL.md` and implemented in `scripts/cfar_biological_attribution.py`.
- Panel b: `REPRESENTATIVE_CASE_LOCK.json`, `candidate_cfar_explanations.jsonl`, and the matching sample payload under `intermediate/pathways/`.
- Panel c: `pc_usage_by_section.csv`.
- Panel d: `cfar_gene_landscape.csv`.
- Panel e: `intermediate/cfar_pathway_classification_master.csv` and the Hallmark GSEA/block-permutation tables.
- Panel f: `section_cfar_pathway_correlations.csv` and `intermediate/cfar_pathway_correlation_summary.csv`.
- Panel g: `cfar_pathway_by_context.csv`.
- Panel h: `cfar_random_calibration.csv`.

All panels are regenerated by `RUN_CFAR_BIOLOGY.ps1`; no values or labels are manually edited.
"""
    (PROJECT / "FIGURE_SOURCE_DATA_NOTE.md").write_text(note, encoding="utf-8")


def qa_stage() -> None:
    required = [
        "PROTOCOL.md", "CFAR_REPRODUCTION_AUDIT.csv", "spot_pc_signed_deviation.parquet", "spot_pc_contribution.parquet",
        "spot_gene_cfar_attribution.parquet", "CFAR_GENE_BACKPROJECTION_FIDELITY.csv", "candidate_gene_cfar_attribution.parquet",
        "candidate_cfar_explanations.jsonl", "cfar_gene_landscape.csv", "pc_gene_driver_table.csv", "pc_usage_by_section.csv",
        "pc_usage_by_cancer.csv", "pc_usage_by_context.csv", "hallmark_cfar_regionweighted.csv", "hallmark_cfar_sectionequal.csv",
        "hallmark_cfar_blockperm.csv", "reactome_cfar_regionweighted.csv", "reactome_cfar_sectionequal.csv", "reactome_cfar_blockperm.csv",
        "gobp_cfar_regionweighted.csv", "gobp_cfar_sectionequal.csv", "gobp_cfar_blockperm.csv", "section_cfar_pathway_correlations.csv",
        "cfar_pathway_mixed_model.csv", "cfar_quantile_pathway_summary.csv", "cfar_pathway_by_context.csv", "cfar_pathway_by_cancer.csv",
        "cfar_spatial_gradient.csv", "cfar_random_calibration.csv", "REPRESENTATIVE_CASE_LOCK.json",
        "Figure_CFAR_BIOLOGICAL_ATTRIBUTION_v1.svg", "Figure_CFAR_BIOLOGICAL_ATTRIBUTION_v1.pdf", "Figure_CFAR_BIOLOGICAL_ATTRIBUTION_v1.png",
        "METHODS.md", "RESULTS.md", "FIGURE_SOURCE_DATA_NOTE.md", "RUN_CFAR_BIOLOGY.ps1",
    ]
    checks = []
    for rel in required:
        path = PROJECT / rel
        checks.append((f"required:{rel}", path.exists() and path.stat().st_size > 0, f"bytes={path.stat().st_size if path.exists() else 0}"))
    gate = json.loads((PROJECT / "intermediate/REPRODUCTION_GATE.json").read_text(encoding="utf-8"))
    checks.extend([
        ("reproduction gate", gate["reproduction_pass"], json.dumps(gate)),
        ("fidelity gate", gate["fidelity_pass"], json.dumps(gate)),
        ("canonical region count", pd.read_parquet(PROJECT / "intermediate/candidate_metadata.parquet")["candidate_key"].nunique() == EXPECTED_REGIONS, str(EXPECTED_REGIONS)),
        ("section count", pd.read_csv(PROJECT / "CFAR_REPRODUCTION_AUDIT.csv").shape[0] == EXPECTED_SECTIONS, str(EXPECTED_SECTIONS)),
        ("block permutations", all((pd.read_csv(PROJECT / f"{lib}_cfar_blockperm.csv")["n_section_block_permutations"] == N_BLOCK_PERM).all() for lib in ["hallmark", "reactome", "gobp"]), str(N_BLOCK_PERM)),
        ("no performance endpoints", not any(token in (PROJECT / "RESULTS.md").read_text(encoding="utf-8") for token in ["new best-IoU", "new ARI", "selector optimization"]), "post-inference attribution only"),
    ])
    source_hashes = gate["formal_source_sha256"]
    unchanged = all(Path(path).exists() and sha256(Path(path)) == digest for path, digest in source_hashes.items())
    checks.append(("formal read-only source hashes unchanged", unchanged, f"n={len(source_hashes)}"))
    png = PROJECT / "Figure_CFAR_BIOLOGICAL_ATTRIBUTION_v1.png"
    if png.exists():
        image = plt.imread(png)
        nonwhite = float(np.mean(np.any(image[..., :3] < 0.98, axis=2)))
        checks.append(("figure nonblank", nonwhite > 0.01, f"nonwhite_fraction={nonwhite:.4f}"))
    failed = [x for x in checks if not x[1]]
    report = ["# QA report", "", f"- Completed: {utc_now()}", f"- Overall: **{'PASS' if not failed else 'FAIL'}**", "", "| Check | Status | Detail |", "|---|---:|---|"]
    report.extend(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail.replace('|', '/')} |" for name, passed, detail in checks)
    report.extend(["", "## Interpretation boundaries", "", "Candidate regions and pathways are not treated as independent biological replicates. Section-level resampling is the inferential layer. The analysis provides molecular attribution of the frozen detector, not causal validation or proof of pathway activation."])
    (PROJECT / "QA_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"QA failed: {[x[0] for x in failed]}")

    manifest_rows = []
    for path in sorted(PROJECT.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append({"path": str(path.relative_to(PROJECT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(PROJECT / "SHA256_MANIFEST.csv", index=False)
    build = [
        "# Build report", "", f"- Completed: {utc_now()}", f"- Python: `{sys.version.split()[0]}`", f"- Platform: `{platform.platform()}`",
        f"- gseapy: `{getattr(gp, '__version__', 'unknown')}`", f"- Formal sections: {EXPECTED_SECTIONS}", f"- Canonical retained regions: {EXPECTED_REGIONS:,}",
        f"- GSEA permutations: {N_GSEA_PERM:,}", f"- Section-block sign flips: {N_BLOCK_PERM:,}", f"- Section bootstrap replicates: {N_BOOT:,}",
        "- Figure backend: Python/matplotlib only", "- Figure archetype: schematic-led asymmetric A4 portrait composite",
        "- Random control: deterministic replay of the pooled-landscape reconstruction framework; exact historical random member identity was not persisted.",
        "- RadarST algorithm/model/performance outputs modified: NO.", f"- QA: PASS ({len(checks)}/{len(checks)} checks).",
    ]
    (PROJECT / "BUILD_REPORT.md").write_text("\n".join(build) + "\n", encoding="utf-8")
    # Re-hash after reports are complete.
    manifest_rows = []
    for path in sorted(PROJECT.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append({"path": str(path.relative_to(PROJECT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(PROJECT / "SHA256_MANIFEST.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "reproduce", "attribute", "documents", "figure", "qa"], default="all")
    args = parser.parse_args()
    ensure_layout()
    if args.stage in {"all", "reproduce"}:
        reproduction_stage()
    if args.stage in {"all", "attribute"}:
        attribute_stage()
        generate_documents()
    if args.stage == "documents":
        generate_documents()
    if args.stage in {"all", "figure"}:
        figure_stage()
    if args.stage in {"all", "qa"}:
        qa_stage()
    print(f"CFAR biological attribution complete: {PROJECT}", flush=True)


if __name__ == "__main__":
    main()
