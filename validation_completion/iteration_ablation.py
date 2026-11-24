"""Exact counterfactual round-budget prefixes of the frozen deterministic policy.
No new model fits, scores, thresholds, candidate pools or reference-driven choices.
"""
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import json,concurrent.futures as cf,argparse,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score
from complete_controls import ROOT,SRC,EXT,WORK,REG,sha,put,paths,qualities,groups_members
OUT=ROOT/'iteration_ablation'
def generate(r):
 inp,pred,_=paths(r);out=OUT/r['cohort']/r['dataset']/r['workflow'];out.mkdir(parents=True,exist_ok=True)
 a=np.load(inp/'MAP.npz',allow_pickle=False);pool=pd.read_parquet(inp/'CANDIDATES.parquet');lk={c:i for i,c in enumerate(pool.candidate_id)}
 anchor=(Path('H:/RadarST_2_ExactMask_20261028/controls')/r['dataset']/r['workflow']/'CLEAN_V1/INSTANCE_REGISTRY.parquet') if r['cohort']=='development' else inp/'CLEAN_FULL_SELECTED.parquet'
 S={lk[c] for c in pd.read_parquet(anchor).candidate_id};d=pd.read_csv(pred/'DECISIONS.csv',keep_default_na=False);hist={'round0':np.asarray(sorted(S),np.int32)}
 for rd in range(6):
  for row in d.loc[d['round'].eq(rd)].itertuples():
   removed={lk[c] for c in row.removed_ids.split(';') if c};assert removed<=S
   S-=removed
   if not row.operation_remove:S.add(lk[row.candidate_id])
  hist['round'+str(rd+1)]=np.asarray(sorted(S),np.int32)
 assert S=={lk[c] for c in pd.read_parquet(pred/'SELECTED_OBJECTS.parquet').candidate_id}
 members,groups,valid=groups_members(pool,a['spots'].astype(str),a['edge']);labels={}
 for name,ids in hist.items():
  used=set();lab=a['base'].copy()
  for i in ids:assert valid[i] and not used.intersection(members[i]);used.update(members[i]);lab[members[i]]=int(a['base'].max())+1+int(i)
  labels[name]=lab
 assert np.array_equal(labels['round6'],pd.read_csv(pred/'FINAL_LABELS.csv').refined.to_numpy())
 np.savez_compressed(out/'PREFIX_LABELS.npz',**labels);np.savez_compressed(out/'PREFIX_SELECTIONS.npz',**hist)
 gate=dict(status='PASS',reference_read=False,exact_final_parity=True,changed_parameter='maximum_rounds_only',grid=list(range(7)),files={str(p):sha(p) for p in [anchor,pred/'DECISIONS.csv',pred/'FINAL_LABELS.csv',out/'PREFIX_LABELS.npz',out/'PREFIX_SELECTIONS.npz']});put(out/'GENERATION_GATE.json',gate);return True
def evaluate(r):
 assert (OUT/'ALL322_PREFIXES_LOCKED.json').exists();inp,pred,_=paths(r);out=OUT/r['cohort']/r['dataset']/r['workflow'];a=np.load(inp/'MAP.npz',allow_pickle=False);spots=a['spots'].astype(str);lk={s:i for i,s in enumerate(spots)}
 t=pd.read_csv(WORK/'PAPER_ALL_COMPONENT_EVENTS.csv');t=t[(t.dataset_id==r['dataset'])&(t.method==r['workflow'])&t.component_size_spots.between(1,10)].reset_index(drop=True);owner=np.full(len(spots),-1,int);sz=t.component_size_spots.to_numpy(int)
 for j,mem in enumerate(t.reference_members):owner[[lk[s] for s in mem.split(';')]]=j
 truth=pd.read_csv(r['reference'],dtype=str,keep_default_na=False).set_index('spot_id').truth_label.reindex(spots).to_numpy(str);known=~np.isin(np.char.lower(truth),['','nan','na','none','null','unknown','<na>','unassigned']);assert known.any()
 lab=np.load(out/'PREFIX_LABELS.npz');selection=np.load(out/'PREFIX_SELECTIONS.npz');pool=pd.read_parquet(inp/'CANDIDATES.parquet');rows=[];events=[]
 for rd in range(7):
  z=lab['round'+str(rd)];ids=selection['round'+str(rd)];q=qualities(z,a['edge'],owner,sz);delta=q-t.base_best_IoU.to_numpy();rec=dict(cohort=r['cohort'],dataset=r['dataset'],workflow=r['workflow'],maximum_rounds=rd,Mean_Best_IoU=float(q.mean()),gain=float(delta.mean()),exact=float((q>=1-1e-12).mean()),near_0_8=float((q>=.8-1e-12).mean()),loss_fraction=float((delta<-1e-12).mean()),ARI=float(adjusted_rand_score(truth[known],z[known])),modified_spots=int(pool.iloc[ids].n_spots.sum()),modified_fraction=float(pool.iloc[ids].n_spots.sum()/len(spots)),candidate_count=len(ids))
  for lo,hi in [(1,2),(3,5),(6,10)]:
   m=(sz>=lo)&(sz<=hi);rec[f'size{lo}_{hi}_Mean']=float(q[m].mean()) if m.any() else np.nan
  rows.append(rec)
  for i,row in t.iterrows():events.append(dict(cohort=r['cohort'],dataset=r['dataset'],workflow=r['workflow'],maximum_rounds=rd,truth_component_id=row.truth_component_id,size=int(sz[i]),best_IoU=float(q[i]),delta_upstream=float(delta[i])))
 actual=pd.read_csv(WORK/'PAPER_ALL322_MAP_METRICS.csv');row=actual[(actual.dataset==r['dataset'])&(actual.workflow==r['workflow'])].iloc[0]
 for key in ['Mean_Best_IoU','exact','near_0_8','ARI','modified_fraction']:assert abs(rows[-1][key]-row[key])<1e-12,(r,key)
 pd.DataFrame(rows).to_csv(out/'RESULTS.csv',index=False);pd.DataFrame(events).to_parquet(out/'COMPONENT_TRANSITIONS.parquet',index=False);return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['generate','evaluate']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();regs=pd.read_csv(REG,keep_default_na=False).to_dict('records');OUT.mkdir(exist_ok=True);start=time.time();rows=[]
 with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
  fs={ex.submit(generate if args.phase=='generate' else evaluate,r):r for r in regs}
  for f in cf.as_completed(fs):
   ans=f.result()
   if args.phase=='evaluate':rows.extend(ans)
   print('ROUND_ABLATION',args.phase,fs[f]['dataset'],fs[f]['workflow'],flush=True)
 if args.phase=='generate':put(OUT/'ALL322_PREFIXES_LOCKED.json',dict(status='LOCKED_BEFORE_NEW_REFERENCE_EVALUATION',maps=322,round_budgets=list(range(7)),gates={str(OUT/r['cohort']/r['dataset']/r['workflow']/'GENERATION_GATE.json'):sha(OUT/r['cohort']/r['dataset']/r['workflow']/'GENERATION_GATE.json') for r in regs},policy_prefix_replay='identical to stopping deterministic frozen inference after that many rounds; no changed score/head/eligibility',code_sha256=sha(Path(__file__))))
 else:
  df=pd.DataFrame(rows);df.to_csv(ROOT/'ITERATION_ABLATION_ALL322_RESULTS.csv',index=False);sec=df.groupby(['cohort','maximum_rounds','dataset']).mean(numeric_only=True).reset_index();sec.to_csv(ROOT/'ITERATION_ABLATION_SECTION_RESULTS.csv',index=False);sec.groupby(['cohort','maximum_rounds']).mean(numeric_only=True).reset_index().to_csv(ROOT/'ITERATION_ABLATION_SUMMARY.csv',index=False)
 put(OUT/f'{args.phase.upper()}_RECEIPT.json',dict(status='PASS',maps=322,complete_grid=list(range(7)),seconds=time.time()-start))
if __name__=='__main__':main()
