from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SEED = 20260912


def stable_seed(*parts: object, base: int = SEED) -> int:
    token = "||".join(map(str, parts)).encode("utf-8")
    return (base + int(hashlib.sha256(token).hexdigest()[:8], 16)) % (2**32 - 1)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    expected = pd.read_csv(args.summary)
    comparisons = []
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    input_hashes = {'summary': sha(args.summary)}
    members_path = args.output / 'random_region_members.csv.gz'
    with members_path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample','candidate_size','replicate','rng_seed','member_spots','median_CFAR'])
            for sample, group in expected.groupby('sample', sort=False):
                path = args.input_dir / f'{sample}.npz'
                input_hashes[sample] = sha(path)
                with np.load(path, allow_pickle=False) as saved:
                    coords, cfar, ids = saved['coords'], saved['cfar'], saved['spot_ids'].astype(str)
                graph = symmetric_knn_graph(coords, k=6)
                for row in group.sort_values('candidate_size').itertuples():
                    size = int(row.candidate_size)
                    seed = stable_seed(sample, size, 'matched_random', base=20260805)
                    rng = np.random.default_rng(seed)
                    regions = []
                    attempts = 0
                    while len(regions) < 500 and attempts < 10000:
                        attempts += 1
                        value = random_connected_set(graph, size, rng)
                        if value is not None:
                            regions.append(value)
                    assert len(regions) == 500
                    values = np.asarray([np.median(cfar[idx]) for idx in regions], dtype=float)
                    for i, (idx, value) in enumerate(zip(regions, values)):
                        writer.writerow([sample,size,i,seed,';'.join(ids[idx]),repr(float(value))])
                    regenerated = float(np.median(values))
                    saved = float(row.random_median_CFAR)
                    comparisons.append({'sample':sample,'candidate_size':size,'n_random':len(regions),
                                        'stored_random_median':saved,'regenerated_random_median':regenerated,
                                        'absolute_difference':abs(regenerated-saved),
                                        'matches_at_1e_8':abs(regenerated-saved)<=1e-8})
                print(sample, len(group), flush=True)
    pd.DataFrame(comparisons).to_csv(args.output/'summary_comparison.csv', index=False)
    result = {'cells':len(comparisons),'controls':len(comparisons)*500,
              'summary_matches':int(sum(x['matches_at_1e_8'] for x in comparisons)),
              'max_absolute_difference':max(x['absolute_difference'] for x in comparisons),
              'members_sha256':sha(members_path),'input_sha256':input_hashes,
              'original_member_identity_proven':False,
              'graph_implementation':'Original k=6 function uses n_neighbors=7 and kneighbors(X=None), returning seven neighbors per spot for these inputs.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}, indent=2))


if __name__=='__main__':
    main()
