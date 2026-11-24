from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import datetime,timezone
import hashlib,math,os,time,json
import numpy as np
import pandas as pd
from scipy import stats,sparse
from sklearn.neighbors import NearestNeighbors

ENGINEERING_PATH_CEILING = 64

def canonical_members_from_ids(members: list[str] | tuple[str, ...]) -> str:
    return ';'.join(sorted(map(str, members)))

def generate_full_path(seed: int, scores: np.ndarray, graph: list[set[int]], spots: list[str]) -> list[dict[str, object]]:
    chosen = {int(seed)}
    frontier = set(graph[int(seed)])
    score_sum = float(scores[seed])
    member_text = canonical_members_from_ids([spots[seed]])
    terminal = not frontier or len(chosen) >= min(len(spots), ENGINEERING_PATH_CEILING)
    initial_stop = 'frontier_empty' if not frontier else 'engineering_ceiling' if terminal else ''
    path = [{'prefix_size': 1, 'member_indices': tuple(sorted(chosen)), 'member_spots': member_text, 'added_spot_id': spots[seed], 'added_spot_anomaly': float(scores[seed]), 'formal_path_proposal_score': float(2.0 * scores[seed]), 'emitted_candidate_score_if_applicable': float(scores[seed]), 'frontier_size_before': len(frontier), 'frontier_size_after': len(frontier), 'step_compute_seconds': 0.0, 'path_terminal_after_prefix': terminal, 'path_stop_reason': initial_stop}]
    ceiling = min(len(spots), ENGINEERING_PATH_CEILING)
    while len(chosen) < ceiling and frontier:
        step_start = time.perf_counter()
        frontier_before = len(frontier)
        best = max(frontier, key=lambda item: scores[item])
        frontier.remove(best)
        chosen.add(best)
        score_sum += float(scores[best])
        for neighbor in graph[best]:
            if neighbor not in chosen:
                frontier.add(neighbor)
        member_indices = tuple(sorted(chosen))
        member_text = canonical_members_from_ids([spots[index] for index in member_indices])
        terminal = not frontier or len(chosen) >= ceiling
        stop_reason = 'frontier_empty' if not frontier else 'engineering_ceiling' if len(chosen) >= ceiling else ''
        path.append({'prefix_size': len(chosen), 'member_indices': member_indices, 'member_spots': member_text, 'added_spot_id': spots[best], 'added_spot_anomaly': float(scores[best]), 'formal_path_proposal_score': float(score_sum / len(chosen) + scores[seed]), 'emitted_candidate_score_if_applicable': float(score_sum / len(chosen) + scores[seed]) if len(chosen) >= 3 else np.nan, 'frontier_size_before': frontier_before, 'frontier_size_after': len(frontier), 'step_compute_seconds': time.perf_counter() - step_start, 'path_terminal_after_prefix': terminal, 'path_stop_reason': stop_reason})
    return path
