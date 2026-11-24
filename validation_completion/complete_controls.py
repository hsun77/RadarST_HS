"""Locked label-blind controls, then full-map evaluation. Never changes RadarST."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,argparse,concurrent.futures as cf
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1]
SRC=Path('H:/RadarST_2_ObjectValue_20261104'); EXT=Path('H:/RadarST_E2_FixedTransfer_20261105')
WORK=Path('H:/RadarST_PROJECT/_manuscript_work_20261105')
REG=Path('H:/RadarST_CleanV1_BANKSY_FULL_20261021/audits/REGISTERED_322_PAIRS.csv')
sys.dont_write_bytecode=True
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def paths(r):
 ds,wf=r['dataset'],r['workflow'];dev=r['cohort']=='development'
 inp=(SRC/('inputs' if dev else 'external_inputs'))/ds/wf
 pred=(SRC/'round2_event_alignment/predictions' if dev else EXT/'external_predictions')/ds/wf/'E2_EVENT_CREDIT'
 return inp,pred,ROOT/'random_controls'/r['cohort']/ds/wf
def groups_members(pool,spots,edge):
 lookup={s:i for i,s in enumerate(spots)};members=[np.array([lookup[s] for s in v.split(';')],dtype=np.int32) for v in pool.member_spots]
 adj=[set() for _ in spots]
 for a,b in edge:adj[a].add(b);adj[b].add(a)
 valid=[]
 for m in members:
  core=set(m);seen={int(m[0])};todo=list(seen)
  while todo:
   for j in adj[todo.pop()]:
    if j in core and j not in seen:seen.add(j);todo.append(j)
  valid.append(1<=len(m)<=10 and len(core)==len(m) and seen==core)
 groups={}
 for i,(t,s) in enumerate(zip(pool.proposal_type,pool.n_spots)):
  if valid[i]:groups.setdefault((str(t),int(s)),[]).append(i)
 return members,{k:np.asarray(v,dtype=np.int32) for k,v in groups.items()},valid
def draw(tokens,groups,members,rng,n,ranking=None):
 # Original rejection + permutation scheme; exact quotas and disjointness.
 # Deterministic score ordering for molecular comparator; random restarts only.
 for restart in range(100):
  used=np.zeros(n,bool);chosen=[];ok=True
  for token in tokens:
   options=groups[token];found=-1
   if ranking is None:
    for trial in range(min(512,max(32,len(options)))):
     i=int(options[rng.integers(len(options))])
     if not used[members[i]].any():found=i;break
    if found<0:
     for i in rng.permutation(options):
      if not used[members[i]].any():found=int(i);break
   else:
    for i in ranking[token]:
     if not used[members[i]].any():found=int(i);break
   if found<0:ok=False;break
   chosen.append(found);used[members[found]]=True
  if ok:return np.asarray(chosen,dtype=np.int32)
  # No quota relaxation: change token order only within the same size.
  bysize={}
  for t in tokens:bysize.setdefault(t[1],[]).append(t)
  tokens=[t for s in sorted(bysize,reverse=True) for t in [bysize[s][j] for j in rng.permutation(len(bysize[s]))]]
 raise RuntimeError('INFEASIBLE_AFTER_100_EXACT_MATCH_RESTARTS')
def generate(r,nrep):
 start=time.time();inp,pred,out=paths(r);out.mkdir(parents=True,exist_ok=True)
 gate=out/'GENERATION_GATE.json'
 if gate.exists():
  g=json.loads(gate.read_text());assert g['replicates']==nrep;return g
 a=np.load(inp/'MAP.npz',allow_pickle=False);spots=a['spots'].astype(str);pool=pd.read_parquet(inp/'CANDIDATES.parquet');sel=pd.read_parquet(pred/'SELECTED_OBJECTS.parquet')
 assert pool.candidate_id.is_unique and len(set(spots))==len(spots)
 members,groups,valid=groups_members(pool,spots,a['edge']);lookup={s:i for i,s in enumerate(pool.candidate_id)}
 original=np.array([lookup[s] for s in sel.candidate_id],dtype=np.int32);assert all(valid[i] for i in original)
 used=np.zeros(len(spots),bool)
 for i in original:assert not used[members[i]].any();used[members[i]]=True
 tokens=[(str(t),int(s)) for t,s in zip(sel.proposal_type,sel.n_spots)];tokens.sort(key=lambda x:(-x[1],x[0]))
 seed=int.from_bytes(hashlib.sha256(f"{r['dataset']}|{r['workflow']}|20260812".encode()).digest()[:8],'little')%(2**32-1)
 rng=np.random.default_rng(seed);draws=np.empty((nrep,len(tokens)),np.int32)
 for j in range(nrep):draws[j]=draw(tokens.copy(),groups,members,rng,len(spots))
 raw=pd.read_parquet(inp/'BASE_FEATURES.parquet');cols=[c for c in raw if 'score_per_spot_percentile' in c];assert len(cols)==1,cols
 score=raw[cols[0]].to_numpy(float);assert len(score)==len(pool) and np.isfinite(score).all()
 ranking={k:sorted(v,key=lambda i:(-score[i],str(pool.candidate_id.iloc[i]))) for k,v in groups.items()}
 molecular=draw(tokens.copy(),groups,members,np.random.default_rng(seed),len(spots),ranking)
 # Each saved row is a complete reconstructible map, not an oracle/mask score.
 np.savez_compressed(out/'LOCKED_MEMBERSHIPS.npz',random_indices=draws,molecular_indices=molecular,original_indices=original)
 g=dict(status='PASS',cohort=r['cohort'],dataset=r['dataset'],workflow=r['workflow'],replicates=nrep,seed=seed,matched_count=len(tokens),matched_area=int(used.sum()),matching_fields=['proposal_type','n_spots','count','actual_candidate_member_area'],channel='NOT_APPLICABLE_TO_FINAL_CONDITIONAL_OBJECTS',eligibility='same frozen pool; static connectivity/size/nonoverlap, not learned dynamic conditional eligibility',generation_reference_reads=False,files={str(p):sha(p) for p in [inp/'MAP.npz',inp/'CANDIDATES.parquet',inp/'BASE_FEATURES.parquet',pred/'SELECTED_OBJECTS.parquet',pred/'FINAL_LABELS.csv',out/'LOCKED_MEMBERSHIPS.npz']},seconds=time.time()-start)
 put(gate,g);return g
def qualities(labels,edge,owner,rsizes):
 n=len(labels);same=labels[edge[:,0]]==labels[edge[:,1]];e=edge[same];g=csr_matrix((np.ones(len(e),np.int8),(e[:,0],e[:,1])),shape=(n,n));nc,cc=connected_components(g,directed=False);ps=np.bincount(cc,minlength=nc)
 m=owner>=0;codes,count=np.unique(owner[m]*nc+cc[m],return_counts=True);ri=codes//nc;pi=codes%nc;q=np.zeros(len(rsizes));np.maximum.at(q,ri,count/(rsizes[ri]+ps[pi]-count));return q
def evaluate(r):
 start=time.time();inp,pred,out=paths(r);gate=out/'EVALUATION_GATE.json'
 if gate.exists():return json.loads(gate.read_text())
 lock=ROOT/'audits/ALL_NULL_PREDICTIONS_LOCKED.json';assert lock.exists();lg=json.loads(lock.read_text());assert lg['maps']==322
 gen=json.loads((out/'GENERATION_GATE.json').read_text());assert sha(out/'LOCKED_MEMBERSHIPS.npz')==gen['files'][str(out/'LOCKED_MEMBERSHIPS.npz')]
 a=np.load(inp/'MAP.npz',allow_pickle=False);spots=a['spots'].astype(str);edge=a['edge'];base=a['base'];pool=pd.read_parquet(inp/'CANDIDATES.parquet');lookup={s:i for i,s in enumerate(spots)}
 members=[np.array([lookup[s] for s in v.split(';')],int) for v in pool.member_spots]
 tab=pd.read_csv(WORK/'PAPER_ALL_COMPONENT_EVENTS.csv');tab=tab.loc[(tab.dataset_id==r['dataset'])&(tab.method==r['workflow'])&tab.component_size_spots.between(1,10)].copy();assert len(tab)>0
 owner=np.full(len(spots),-1,int);sizes=tab.component_size_spots.to_numpy(int)
 for j,m in enumerate(tab.reference_members):
  ids=[lookup[s] for s in m.split(';')];assert (owner[ids]<0).all();owner[ids]=j
 # Reference connectivity may differ from inference connectivity for HER2ST.
 # frozen reference memberships already capture source-specific definition;
 # predicted components must use the same evaluation graph as frozen evaluator.
 # HER2ST array graph defines REFERENCES only. Prediction connectivity stays
 # on inference physical coordinates, as in the frozen ObjectEvaluator.
 def labels(indices):
  z=base.copy()
  for i in indices:z[members[i]]=int(base.max())+1+int(i)
  return z
 saved=np.load(out/'LOCKED_MEMBERSHIPS.npz');q0=qualities(base,edge,owner,sizes);actual=pd.read_csv(pred/'FINAL_LABELS.csv').refined.to_numpy();qa=qualities(actual,edge,owner,sizes)
 assert np.allclose(q0,tab.base_best_IoU.to_numpy(),atol=1e-12,rtol=0)
 assert np.allclose(qa,tab.after_best_IoU.to_numpy(),atol=1e-12,rtol=0)
 # Full-source independent evaluator parity on one null and actual.
 sys.path.insert(0,str(SRC/'code'));from object_evidence import adjacency
 from object_metrics import ObjectEvaluator
 from external_fixed_policy import reference_evaluator
 truth=pd.read_csv(r['reference'],dtype=str,keep_default_na=False).set_index('spot_id').truth_label.reindex(spots);adj=adjacency(len(spots),a['edge'])
 ev=ObjectEvaluator(truth.to_numpy(),adj) if r['cohort']=='development' else reference_evaluator(r,list(spots),a['xy'],truth.to_numpy(),adj)[0]
 qtest=ev.evaluate(labels(saved['random_indices'][0]),True)[1][ev.sizes<=10]
 # Component ordering in table derives from exact original evaluator.
 assert np.allclose(qualities(labels(saved['random_indices'][0]),edge,owner,sizes),qtest,rtol=0,atol=1e-12)
 rows=[]
 for method,seq in [('matched_random',saved['random_indices']),('molecular_order',[saved['molecular_indices']]),('RadarST',[saved['original_indices']])]:
  for j,inds in enumerate(seq):
   z=labels(inds);q=qualities(z,edge,owner,sizes);delta=q-q0
   rec=dict(cohort=r['cohort'],dataset=r['dataset'],workflow=r['workflow'],control=method,replicate=j,Mean_Best_IoU=float(q.mean()),gain=float(delta.mean()),exact=float((q>=1-1e-12).mean()),near_0_8=float((q>=.8-1e-12).mean()),loss_fraction=float((delta<-1e-12).mean()),modified_spots=gen['matched_area'],candidate_count=len(inds))
   for lo,hi in [(1,2),(3,5),(6,10)]:
    m=(sizes>=lo)&(sizes<=hi);rec[f'size{lo}_{hi}_Mean']=float(q[m].mean()) if m.any() else np.nan
   rows.append(rec)
 d=pd.DataFrame(rows);d.to_csv(out/'COMPLETE_MAP_RESULTS.csv',index=False)
 g=dict(status='PASS',dataset=r['dataset'],workflow=r['workflow'],rows=len(d),independent_evaluator_max_error=0,observed_frozen_max_error=0,seconds=time.time()-start,results_sha256=sha(out/'COMPLETE_MAP_RESULTS.csv'));put(gate,g);return g
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['generate','evaluate']);ap.add_argument('--replicates',type=int,default=1000);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 regs=pd.read_csv(REG,keep_default_na=False).to_dict('records');assert len(regs)==322
 if args.limit:regs=regs[:args.limit]
 start=time.time();success=[];fail=[]
 with cf.ProcessPoolExecutor(max_workers=args.workers) as exe:
  fs={exe.submit(generate,r,args.replicates) if args.phase=='generate' else exe.submit(evaluate,r):r for r in regs}
  for future in cf.as_completed(fs):
   r=fs[future]
   try:
    ans=future.result();success.append(ans);print(args.phase,'PASS',r['cohort'],r['dataset'],r['workflow'],round(ans['seconds'],2),flush=True)
   except Exception as e:
    import traceback;fail.append(dict(dataset=r['dataset'],workflow=r['workflow'],error=repr(e),traceback=traceback.format_exc()));print(args.phase,'FAIL',r['dataset'],r['workflow'],repr(e),flush=True)
 put(ROOT/'audits'/f'{args.phase.upper()}_RUN_RECEIPT.json',dict(success=len(success),failed=fail,seconds=time.time()-start,workers=args.workers))
 if fail:raise RuntimeError(f'{len(fail)} failures; preserved, no relaxation')
 if args.phase=='generate' and not args.limit:
  put(ROOT/'audits/ALL_NULL_PREDICTIONS_LOCKED.json',dict(status='LOCKED_BEFORE_REFERENCE_EVALUATION',maps=322,replicates=args.replicates,gates={str(paths(r)[2]/'GENERATION_GATE.json'):sha(paths(r)[2]/'GENERATION_GATE.json') for r in regs},code_sha256=sha(Path(__file__))))
 if args.phase=='evaluate' and not args.limit:
  data=pd.concat([pd.read_csv(paths(r)[2]/'COMPLETE_MAP_RESULTS.csv') for r in regs],ignore_index=True);data.to_csv(ROOT/'MATCHED_CONTROL_ALL322_RESULTS.csv',index=False)
  summary=[]
  for cohort,g in data.groupby('cohort'):
   sec=g.groupby(['control','replicate','dataset']).mean(numeric_only=True).reset_index();agg=sec.groupby(['control','replicate']).mean(numeric_only=True);null=agg.loc['matched_random'];obs=agg.loc['RadarST'].iloc[0];mol=agg.loc['molecular_order'].iloc[0]
   for metric in ['Mean_Best_IoU','gain','exact','near_0_8','loss_fraction','size3_5_Mean']:
    nv=null[metric].dropna().to_numpy();summary.append(dict(cohort=cohort,metric=metric,RadarST=float(obs[metric]),molecular_order=float(mol[metric]),random_mean=float(nv.mean()),random_interval_low=float(np.quantile(nv,.025)),random_interval_high=float(np.quantile(nv,.975)),empirical_upper_p=float((1+(nv>=obs[metric]-1e-12).sum())/(1+len(nv))),replicates=len(nv)))
  pd.DataFrame(summary).to_csv(ROOT/'MATCHED_CONTROL_COHORT_SUMMARY.csv',index=False)
 print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
