from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter,deque
from datetime import datetime,timezone
import hashlib,math,os,time,json
import numpy as np
import pandas as pd
from scipy import stats,sparse
from sklearn.neighbors import NearestNeighbors

SEED = 20260812
N_RANDOM = 1000
N_BOOTSTRAP = 10000
SMALL_MAX = 10

def exact_sampler(frame: pd.DataFrame, final_pair: pd.DataFrame, spot_ids: list[str]):
    records = H.candidate_records(frame, spot_ids)
    (core_mask, singleton_mask, tail_mask, route) = H.eligibility_masks(frame)
    pools: dict[str, dict[tuple[str, int], list[dict]]] = {'core': defaultdict(list), 'singleton': defaultdict(list), 'tail': defaultdict(list)}
    spot_to_index = {str(spot): i for (i, spot) in enumerate(spot_ids)}
    for record in records:
        idx = int(record['index'])
        record['spot_indices'] = np.asarray([spot_to_index[s] for s in record['members']], dtype=np.int32)
        key = (str(record['proposal_type']), int(record['n_spots']))
        if core_mask[idx]:
            pools['core'][key].append(record)
        if singleton_mask[idx]:
            pools['singleton'][key].append(record)
        if tail_mask[idx]:
            pools['tail'][key].append(record)
    tokens = [(str(row.channel), str(row.proposal_type), int(row.n_spots)) for row in final_pair.itertuples(index=False)]
    tokens.sort(key=lambda x: (-x[2], x[0], x[1]))
    availability = Counter()
    for (channel, typed) in pools.items():
        for ((proposal_type, size), options) in typed.items():
            availability[channel, proposal_type, size] = len(options)
    needed = Counter(tokens)
    missing = {str(k): v for (k, v) in needed.items() if availability[k] == 0}
    if missing:
        raise RuntimeError(f'Exact final token types unavailable: {missing}')
    return {'pools': pools, 'tokens': tokens, 'route': route, 'availability': availability}

def draw_records(sampler: dict, rng: np.random.Generator, max_restarts: int=100):
    for _ in range(max_restarts):
        touched: set[str] = set()
        chosen_records = []
        for (channel, proposal_type, size) in sampler['tokens']:
            options = sampler['pools'][channel][proposal_type, size]
            chosen = None
            n_trials = min(512, max(32, len(options)))
            for _trial in range(n_trials):
                candidate = options[int(rng.integers(0, len(options)))]
                if not candidate['members'] & touched:
                    chosen = candidate
                    break
            if chosen is None:
                for pos in rng.permutation(len(options)):
                    candidate = options[int(pos)]
                    if not candidate['members'] & touched:
                        chosen = candidate
                        break
            if chosen is None:
                break
            chosen_records.append(chosen)
            touched.update(chosen['members'])
        if len(chosen_records) == len(sampler['tokens']):
            return (chosen_records, len(touched))
    raise RuntimeError('Unable to sample an exact non-overlapping final-token set')

def random_small_gain(state: dict, base_matches: dict, small_ids: list[str], records: list[dict]):
    augmented = state['base_codes'].copy()
    next_code = int(augmented.max()) + 1
    for (offset, record) in enumerate(records):
        augmented[record['spot_indices']] = next_code + offset
    (component_codes, component_sizes) = H.connected_match_state(augmented, state['edge_i'], state['edge_j'])
    deltas = []
    for component_id in small_ids:
        after = H.best_match_fast(component_codes, component_sizes, state['truth_indices'][component_id])
        deltas.append(float(after['iou'] - base_matches[component_id]['iou']))
    return float(np.mean(deltas)) if deltas else 0.0

def pair_seed(dataset_id: str, method: str, base_seed: int) -> int:
    import hashlib
    digest = hashlib.sha256(f'{dataset_id}|{method}|{base_seed}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32 - 1)
