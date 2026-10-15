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


EPS = 1e-12

def canonical_spots(value: str) -> str:
    return ";".join(sorted(set(B67.parse_spots(str(value)))))

def candidate_key(dataset: str, proposal_type: str, member_spots: str) -> str:
    return f"{dataset}||{proposal_type}||{canonical_spots(member_spots)}"

def load_teacher_keys() -> set[tuple[str, str, str]]:
    teacher = pd.read_csv(TEACHER, low_memory=False)
    numeric = ["fragmentation_risk", "erosion_risk", "boundary_leakage_risk", "evidence_score"]
    for column in numeric:
        teacher[column] = pd.to_numeric(teacher[column], errors="coerce")
    nonfragmenting = teacher["fragmentation_risk"].le(EPS) & teacher["erosion_risk"].le(1 / 3 + EPS)
    safe_fragment = (
        teacher["fragmentation_risk"].gt(EPS)
        & teacher["erosion_risk"].le(0.05 + EPS)
        & teacher["boundary_leakage_risk"].le(1 / 3 + EPS)
        & teacher["evidence_score"].ge(0.65 - EPS)
    )
    teacher = teacher[nonfragmenting | safe_fragment].copy()
    return {
        (
            str(row.dataset_id),
            str(row.method),
            candidate_key(str(row.dataset_id), str(row.proposal_type), str(row.member_spots)),
        )
        for row in teacher.itertuples(index=False)
    }
