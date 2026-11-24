"""Independent spot/candidate identity checks; never opens reference labels."""
import sys,json,time
sys.path.insert(0,'H:/RadarST_FigureRevision2_20261111')
from sensitivity import registry,ROOT,W,sha,put
import numpy as np,pandas as pd
rows=[];start=time.time()
for r in registry():
 p=ROOT/'inputs'/r['dataset']/r['workflow']
 with np.load(p/'MAP.npz',allow_pickle=False) as z:spots=z['spots'].astype(str)
 ids=pd.read_parquet(p/'CANDIDATES.parquet',columns=['candidate_id']).candidate_id
 assert len(np.unique(spots))==len(spots),(r['dataset'],r['workflow'],'DUPLICATE_SPOT')
 assert ids.notna().all() and ids.is_unique,(r['dataset'],r['workflow'],'DUPLICATE_CANDIDATE')
 rows.append(dict(dataset=r['dataset'],workflow=r['workflow'],spots=len(spots),candidates=len(ids),unique_spots=True,unique_candidates=True))
assert len(rows)==168
put(W/'qa/ALL168_INPUT_IDENTITY_AUDIT.json',dict(status='PASS',maps=len(rows),candidate_events=sum(r['candidates'] for r in rows),reference_files_read=False,seconds=time.time()-start,code_sha256=sha(__file__),rows=rows))
print('INPUT_IDENTITIES_PASS',len(rows),sum(r['candidates'] for r in rows))
