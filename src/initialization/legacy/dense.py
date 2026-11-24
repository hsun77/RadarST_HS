from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import datetime,timezone
import hashlib,math,os,time,json
import numpy as np
import pandas as pd
from scipy import stats,sparse
from sklearn.neighbors import NearestNeighbors

RETENTION = 0.97
EPS = 1e-12
MASTER_SEED = 20260811
ATTEMPTS = 96
BLOCK_WIDTH = 32
RANK_DECAY = 5.0
CHANNEL_ORDER = ('tail', 'core', 'singleton')

def channel_for_formal(row) -> str:
    if str(row.stage) == 'additive_signed_utility_tail':
        return 'tail'
    if str(row.proposal_type) == 'singleton':
        return 'singleton'
    return 'core'

def formal_targets(formal_rows: pd.DataFrame) -> dict[str, int]:
    targets = {channel: 0 for channel in CHANNEL_ORDER}
    for row in formal_rows.itertuples(index=False):
        targets[channel_for_formal(row)] += int(row.n_spots)
    targets['total'] = int(sum(targets.values()))
    return targets

def hash_uniform(candidate_key: str, attempt: int, channel: str) -> float:
    payload = f'{MASTER_SEED}|{attempt}|{channel}|{candidate_key}'.encode('utf-8')
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')
    return (value + 0.5) / float(2 ** 64)

def ordered_blocks(options: list[dict[str, object]], attempt: int):
    if attempt == 0:
        yield from options
        return
    for start in range(0, len(options), BLOCK_WIDTH):
        block = options[start:start + BLOCK_WIDTH]
        priorities = []
        for (rank, item) in enumerate(block):
            u = min(max(hash_uniform(str(item['candidate_key']), attempt, str(item['channel'])), 1e-15), 1 - 1e-15)
            gumbel = -math.log(-math.log(u))
            priorities.append((-rank / RANK_DECAY + gumbel, str(item['candidate_key']), item))
        priorities.sort(key=lambda value: (-value[0], value[1]))
        for (_priority, _key, item) in priorities:
            yield item

def arithmetic_reachability(target: int, sizes: set[int]) -> np.ndarray:
    reachable = np.zeros(target + 1, dtype=bool)
    reachable[0] = True
    for amount in range(1, target + 1):
        reachable[amount] = any((amount >= size and reachable[amount - size] for size in sizes))
    return reachable

def relaxed_assignment(module, frame: pd.DataFrame, combined: np.ndarray, formal_rows: pd.DataFrame, spot_ids: list[str], method: str, variant: str) -> tuple[pd.DataFrame, dict[str, object]]:
    records = module.R.candidate_records(frame, spot_ids)
    (core_mask, singleton_mask, tail_mask, route) = module.R.eligibility_masks(frame)
    masks = {'core': core_mask, 'singleton': singleton_mask, 'tail': tail_mask}
    targets = formal_targets(formal_rows)
    pools: dict[str, list[dict[str, object]]] = {channel: [] for channel in CHANNEL_ORDER}
    for record in records:
        index = int(record['index'])
        for channel in CHANNEL_ORDER:
            if bool(masks[channel][index]):
                item = dict(record)
                item['channel'] = channel
                item['score'] = float(combined[index])
                pools[channel].append(item)
    for channel in CHANNEL_ORDER:
        pools[channel].sort(key=lambda item: (-float(item['score']), str(item['candidate_key'])))
    best: tuple | None = None
    attempt_rows: list[dict[str, object]] = []
    for attempt in range(ATTEMPTS):
        touched: set[str] = set()
        selected: list[dict[str, object]] = []
        achieved = {channel: 0 for channel in CHANNEL_ORDER}
        weighted_score = 0.0
        for channel in CHANNEL_ORDER:
            target = int(targets[channel])
            remaining = target
            sizes = {int(item['n_spots']) for item in pools[channel] if int(item['n_spots']) <= target}
            reachable = arithmetic_reachability(target, sizes) if sizes else np.zeros(target + 1, dtype=bool)
            for item in ordered_blocks(pools[channel], attempt):
                size = int(item['n_spots'])
                if size > remaining or item['members'] & touched:
                    continue
                remainder = remaining - size
                if remainder >= 0 and len(reachable) > remainder and (not reachable[remainder]):
                    continue
                selected.append(item)
                touched.update(item['members'])
                achieved[channel] += size
                remaining -= size
                weighted_score += size * float(item['score'])
                if remaining == 0:
                    break
        shortfall = {channel: int(targets[channel] - achieved[channel]) for channel in CHANNEL_ORDER}
        total_shortfall = int(sum(shortfall.values()))
        signature = '|'.join(sorted((str(item['candidate_key']) for item in selected)))
        objective = (total_shortfall, -weighted_score, signature)
        attempt_rows.append({'attempt': attempt, 'total_shortfall': total_shortfall, 'spot_weighted_score': weighted_score, **{f'achieved_{channel}_spots': achieved[channel] for channel in CHANNEL_ORDER}, **{f'shortfall_{channel}_spots': shortfall[channel] for channel in CHANNEL_ORDER}})
        if best is None or objective < best[0]:
            best = (objective, selected, achieved, attempt, weighted_score)
    if best is None:
        raise RuntimeError(f'No relaxed attempt was generated: {variant} {method}')
    (_objective, selected, achieved, selected_attempt, weighted_score) = best
    rows = []
    for (number, item) in enumerate(selected, start=1):
        row = frame.iloc[int(item['index'])]
        rows.append({'dataset_id': str(row.dataset_id), 'sample_id': str(row.sample_id), 'method': method, 'variant': variant, 'candidate_id': str(item['candidate_id']), 'candidate_key': str(item['candidate_key']), 'proposal_type': str(item['proposal_type']), 'member_spots': ';'.join(sorted(map(str, item['members']))), 'n_spots': int(item['n_spots']), 'channel': str(item['channel']), 'selection_score': float(item['score']), 'topology_valid': bool(row.topology_valid), 'fragmentation_risk': float(row.fragmentation_risk), 'erosion_risk': float(row.erosion_risk), 'boundary_leakage_risk': float(row.boundary_leakage_risk), 'cross_component_risk': float(row.cross_component_risk), 'expression_percentile': float(row.score_per_spot_percentile), 'search_attempt_selected': int(selected_attempt), 'selection_order_pre_retention': number})
    accepted = pd.DataFrame(rows)
    audit = {'safety_route': route, 'search_attempt_selected': int(selected_attempt), 'search_attempts': ATTEMPTS, 'spot_weighted_score': float(weighted_score), **{f'target_{channel}_spots': int(targets[channel]) for channel in CHANNEL_ORDER}, **{f'achieved_{channel}_spots': int(achieved[channel]) for channel in CHANNEL_ORDER}, **{f'absolute_{channel}_difference': int(abs(targets[channel] - achieved[channel])) for channel in CHANNEL_ORDER}, 'target_total_spots': int(targets['total']), 'achieved_total_spots': int(sum(achieved.values())), 'absolute_total_difference': int(abs(targets['total'] - sum(achieved.values()))), 'exact_channel_burden_match': bool(all((targets[channel] == achieved[channel] for channel in CHANNEL_ORDER))), 'method_name_used_in_random_seed': False, 'variant_name_used_in_random_seed': False, 'common_random_number_key': 'master_seed|attempt|channel|candidate_key', 'attempt_diagnostics': attempt_rows}
    return (accepted, audit)

def retained_selection(accepted: pd.DataFrame, formal_total: int, method: str, variant: str) -> tuple[dict[str, str], pd.DataFrame, int, int]:
    target = int(np.floor(float(formal_total) * RETENTION + EPS))
    if accepted.empty:
        return ({}, accepted.copy(), 0, target)
    ordered = accepted.sort_values(['selection_score', 'candidate_key'], ascending=[False, True]).copy()
    keep_indices: list[int] = []
    used = 0
    for (index, row) in ordered.iterrows():
        size = int(row.n_spots)
        if used + size > target:
            continue
        keep_indices.append(index)
        used += size
    retained = accepted.loc[keep_indices].copy()
    retained = retained.sort_values(['selection_score', 'candidate_key'], ascending=[False, True]).reset_index(drop=True)
    retained['selection_order_post_retention'] = np.arange(1, len(retained) + 1)
    assignment: dict[str, str] = {}
    for (number, row) in enumerate(retained.itertuples(index=False), start=1):
        label = f'RadarST_dense_selector__{variant}__{method}__{number}'
        for spot in str(row.member_spots).split(';'):
            if spot:
                if spot in assignment:
                    raise RuntimeError(f'Post-retention overlap: {variant} {method} {spot}')
                assignment[spot] = label
    return (assignment, retained, used, target)
