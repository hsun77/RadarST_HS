"""Prelocked 5x6 initialization sensitivity; frozen LOSO heads and E2 policy."""
import os,sys,time,json,hashlib,argparse,math,collections
from pathlib import Path
sys.dont_write_bytecode=True
W=Path('H:/RadarST_FigureRevision2_20261111');ROOT=Path('H:/RadarST_2_ObjectValue_20261104');OLD=Path('H:/RadarST_I3_REBUILD_20261021');CLEAN=OLD/'frozen_assets/clean_runtime'
for p in ['before','figures','qa','data','staged','tmp','predictions','evaluation']: (W/p).mkdir(parents=True,exist_ok=True)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
for k in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(W/'tmp')
import numpy as np,pandas as pd
REG=Path('H:/RadarST_CleanV1_BANKSY_FULL_20261021/audits/REGISTERED_322_PAIRS.csv')
CONFIGS=[dict(id=f'w{int(round(w*100)):02d}_r{int(round(r*100)):03d}',expression_weight=w,expected_gain_weight=round(1-w,2),retention=r) for w in [.70,.75,.80,.85,.90] for r in [.95,.96,.97,.98,.99,1.0]]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def put(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v:v.item()),encoding='utf-8');os.replace(q,p)
def registry():return pd.read_csv(REG,keep_default_na=False).query('cohort=="development"').sort_values(['dataset','workflow']).to_dict('records')
def configure_modules(inference=True):
 sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(CLEAN/'src'))
 import common
 # Old modules set temporary directories at import; redirect these immediately.
 for k in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(W/'tmp')
 if inference:
  refs={str(Path(r['reference']).resolve()).lower() for r in registry()};refs|={str(Path(r['metadata']).resolve()).lower() for r in registry() if r.get('metadata')}
  def hook(event,args):
   if event!='open' or not isinstance(args[0],(str,bytes)):return
   raw=os.fsdecode(args[0])
   if raw.lower()=='nul':return
   p=Path(raw).resolve();mode,flags=args[1:3];write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if write:
    if not p.is_relative_to(W.resolve()):raise RuntimeError('WRITE_OUTSIDE_SENSITIVITY '+str(p))
   elif str(p).lower() in refs or any(p.is_relative_to(q) for q in [ROOT/'targets',ROOT/'evaluation',ROOT/'round2_event_alignment/targets',ROOT/'round2_event_alignment/evaluation',Path('H:/RadarST_PROJECT/_manuscript_work_20261105')]):raise RuntimeError('REFERENCE_OR_ENDPOINT_IN_PREDICTION '+str(p))
  sys.addaudithook(hook)
 return common
def models(ds):
 import infer_event_round2 as engine
 u,e=engine.bundles(ds)
 assert ds not in u['training_datasets'] and ds not in e['training_datasets']
 for v in [u,e]:v['model'].set_params(n_jobs=1)
 assert u['columns']==e['columns'];return u,e
def load_inputs(ds,wf):
 from object_evidence import assemble,gene_evidence
 p=ROOT/'inputs'/ds/wf;assert (p/'INPUT_GATE.json').exists()
 with np.load(p/'MAP.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 f=pd.read_parquet(p/'CANDIDATES.parquet');x=pd.read_parquet(p/'BASE_FEATURES.parquet');c=assemble(a,f,x)
 assert (ROOT/'features'/ds/'GENE_OBJECT_EVIDENCE.parquet').exists()
 genes=gene_evidence(c,ds)
 front=OLD/'runs/a0_frontend'/ds/wf;gate=json.loads((front/'FRONTEND.json').read_text(encoding='utf-8'))
 assert gate['baseline_model_route']==ds and gate['no_reference_read']
 for key in ['full_features','upstream']:
  v=gate['files'][key];assert sha(v['path'])==v['sha256']
 frame=pd.read_parquet(front/'features.parquet');ranking=pd.read_parquet(front/'ranking.parquet');slots=pd.read_csv(front/'slots.csv')
 assert frame.candidate_id.tolist()==ranking.candidate_id.tolist();assert set(frame.candidate_id)==set(c['ids']);assert frame.candidate_id.is_unique
 assert not any('truth' in k.lower() or k.lower().startswith('actual_') for k in frame.columns)
 assert set(np.unique(a['spots'].astype(str)))==set(a['spots'].astype(str))
 return c,genes,frame,ranking,slots,front
def initial_sets(c,frame,ranking,slots,wf,configs):
 from radarst import core
 f=core.functions();f.sel.N_SEARCH_ATTEMPTS=96
 result={};search=[]
 for w in sorted({q['expression_weight'] for q in configs}):
  score=w*frame.score_per_spot_percentile.to_numpy()+round(1-w,2)*ranking.score_expected_gain_rank.to_numpy()-.025*frame.cross_component_risk.to_numpy()
  _,selected,sg=f.sel.exact_token_assignment(frame,score,score,slots,c['a']['spots'].astype(str).tolist(),wf,'anchor_cross_0.025')
  for config in [q for q in configs if q['expression_weight']==w]:
   budget=math.floor(config['retention']*selected.n_spots.sum()+1e-12);chosen=[];used=0
   for row in selected.sort_values(['selection_score','candidate_key'],ascending=[False,True]).itertuples():
    if used+row.n_spots<=budget:chosen.append(row.candidate_id);used+=row.n_spots
   result[config['id']]={c['lookup'][cid] for cid in chosen};search.append(dict(config=config['id'],pre_retention_count=len(selected),retained_count=len(chosen),retained_spots=int(used),budget=int(budget),**{k:v for k,v in sg.items() if isinstance(v,(int,float,str,bool))}))
 return result,search
def refine(c,genes,S0,u,e,cache):
 from fast_infer_round import checked
 from object_evidence import labels_from
 ids=c['ids'];members=c['members'];n=len(c['a']['spots']);indices=c['legal_indices']
 # Legal pool identities are validated once per input before all configurations.
 S=S0.copy();seen=set();credit=0.;trace=[];hits=0;fresh=0
 for rnd in range(6):
  key=tuple(sorted(S))
  if key in seen:break
  seen.add(key)
  if key in cache:pr,pe,feet,conf=cache[key];cache.move_to_end(key);hits+=1
  else:
   dyn,feet,conf=checked(c,S,indices,[int(i in S) for i in indices],True);x=c['x'].iloc[indices].reset_index(drop=True).copy()
   for k in dyn:x[k]=dyn[k].to_numpy()
   for k in genes:x[k]=genes.iloc[indices][k].to_numpy()
   xx=x[u['columns']].to_numpy(np.float32);assert np.isfinite(xx).all();pr=u['model'].predict(xx)*u['scale'];pr[:,:6]=np.maximum(pr[:,:6],0);pe=e['model'].predict(xx)*e['scale'];assert pe.shape==(len(indices),3)
   cache[key]=(pr,pe,feet,conf);fresh+=1
   if len(cache)>24:cache.popitem(last=False)
  net=pr[:,:3]-pr[:,3:6];score=net.sum(1);events=pe.sum(1);allowed=(score>1e-12)&(net>=0).all(1)&(pr[:,6:9].sum(1)>=0)&(pr[:,9]>=0)
  order=sorted(np.flatnonzero(allowed),key=lambda k:((events[k]>1e-12),-score[k],ids[indices[k]]));touched=set();changed=0
  for k in order:
   i=indices[k];foot=feet[k];closed=foot|{j for v in foot for j in c['adj'][v]}
   if touched&closed or not conf[k]<=S or events[k]>credit+1e-12:continue
   removal=i in S;old=conf[k];prior=credit;S-=old
   if not removal:S.add(i)
   touched|=closed;changed+=1;credit-=float(events[k]);assert credit>=-1e-10
   trace.append(dict(round=rnd,candidate_id=ids[i],operation_remove=removal,removed_ids=[ids[j] for j in sorted(old)],predicted_net=float(score[k]),predicted_event_delta=float(events[k]),expected_credit_before=prior,expected_credit_after=credit))
  if not changed:break
 labels=labels_from(S,members,c['a']['base']);return labels,S,trace,dict(rounds=rnd+1,cached_states=hits,new_states=fresh,credit=credit)
def predict_one(reg,u,e,formal_only=False):
 ds,wf=reg['dataset'],reg['workflow'];out=W/('engineering' if formal_only else 'predictions')/ds/wf;gatepath=out/'PREDICTION_GATE.json'
 if gatepath.exists():
  g=json.loads(gatepath.read_text());assert sha(out/'FINAL_MAPS.npz')==g['predictions_sha256'];return g
 start=time.time();c,genes,frame,ranking,slots,front=load_inputs(ds,wf);configs=[q for q in CONFIGS if q['id']=='w80_r097'] if formal_only else CONFIGS
 legal=[]
 for m in c['members']:
  s=set(map(int,m));found={int(m[0])};todo=list(found)
  while todo:
   for j in c['adj'][todo.pop()]:
    if j in s and j not in found:found.add(j);todo.append(j)
  legal.append(found==s and len(s)==len(m) and 1<=len(s)<=10)
 c['legal_indices']=np.flatnonzero(legal).tolist()
 tinit=time.time();anchors,search=initial_sets(c,frame,ranking,slots,wf,configs);initialization_seconds=time.time()-tinit
 frozen_init=pd.read_parquet(Path('H:/RadarST_2_ExactMask_20261028/controls')/ds/wf/'CLEAN_V1/INSTANCE_REGISTRY.parquet');assert anchors['w80_r097']=={c['lookup'][cid] for cid in frozen_init.candidate_id},('INITIALIZATION_PARITY',ds,wf)
 cache=collections.OrderedDict();labels=[];selected=[];init=[];traces={};audit=[]
 for q in configs:
  label,S,trace,info=refine(c,genes,anchors[q['id']],u,e,cache);labels.append(label);selected.append(sorted(S));init.append(sorted(anchors[q['id']]));traces[q['id']]=trace;audit.append(dict(config=q['id'],selected=len(S),area=sum(len(c['members'][i]) for i in S),**info))
  if q['id']=='w80_r097':
   existing=pd.read_csv(ROOT/'round2_event_alignment/predictions'/ds/wf/'E2_EVENT_CREDIT/FINAL_LABELS.csv');assert existing.spot_id.tolist()==c['a']['spots'].astype(str).tolist();assert np.array_equal(label,existing.refined.to_numpy()),('FINAL_POLICY_PARITY',ds,wf)
 out.mkdir(parents=True,exist_ok=True)
 def flatten(v):return np.asarray([x for seq in v for x in seq],np.int32),np.r_[0,np.cumsum([len(seq) for seq in v])]
 selected_values,selected_offsets=flatten(selected);init_values,init_offsets=flatten(init)
 np.savez_compressed(out/'FINAL_MAPS.npz',labels=np.asarray(labels,np.int32),spots=c['a']['spots'].astype(str),config_ids=np.asarray([q['id'] for q in configs]),selected_indices=selected_values,selected_offsets=selected_offsets,initial_indices=init_values,initial_offsets=init_offsets)
 put(out/'DECISIONS.json',traces);pd.DataFrame(audit).to_csv(out/'CONFIGURATION_AUDIT.csv',index=False);pd.DataFrame(search).to_csv(out/'INITIALIZATION_SEARCH.csv',index=False)
 g=dict(status='PASS',dataset=ds,workflow=wf,configurations=len(configs),no_reference_read=True,formal_initialization_exact_parity=True,formal_E2_labels_exact_parity=True,utility_model_sha256=u['sha256'],event_model_sha256=e['sha256'],training_datasets=u['training_datasets'],predictions_sha256=sha(out/'FINAL_MAPS.npz'),seconds=time.time()-start,initialization_seconds=initialization_seconds,new_feature_states=sum(x['new_states'] for x in audit),cached_feature_states=sum(x['cached_states'] for x in audit),input_sha256={str(p):sha(p) for p in [ROOT/'inputs'/ds/wf/'MAP.npz',ROOT/'inputs'/ds/wf/'CANDIDATES.parquet',ROOT/'inputs'/ds/wf/'BASE_FEATURES.parquet',front/'features.parquet',front/'ranking.parquet',front/'slots.csv',ROOT/'features'/ds/'GENE_OBJECT_EVIDENCE.parquet']})
 put(gatepath,g);print('PREDICTED',ds,wf,len(configs),round(g['seconds'],2),flush=True);return g
def worker(ds,workflow=None,formal_only=False):
 lock=W/'IMPLEMENTATION_LOCK.json'
 if lock.exists():assert sha(Path(__file__))==json.loads(lock.read_text())['sensitivity_code_sha256']
 configure_modules(True)
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=1)
 u,e=models(ds);rows=[]
 for r in registry():
  if r['dataset']==ds and (workflow is None or r['workflow']==workflow):rows.append(predict_one(r,u,e,formal_only))
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['lock','engineering','one','predict']);ap.add_argument('--dataset',default='ST_PDAC_B');ap.add_argument('--workflow',default='GraphST');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
 if args.phase=='lock':
  p=W/'PROTOCOL_LOCK.json';assert not p.exists();put(p,dict(task='Original Figure5b 30-configuration sensitivity',configs=CONFIGS,development_maps=168,external_maps=0,maximum_refinement_rounds=6,cross_component_penalty=.025,search_attempts=96,models='Existing frozen section-excluded utility/event heads and initialization outputs; no fit',selection_of_best_config=False,formal_configuration='w80_r097',primary_aggregation='Within component, then workflow, then equal sections; all168 pairs',prediction_lock_before_reference_evaluation=True,no_zero_edit_map_removed_from_statistics=True));print('PROTOCOL_LOCKED');return
 assert (W/'PROTOCOL_LOCK.json').exists()
 if args.phase=='engineering':worker(args.dataset,args.workflow,True);return
 if args.phase=='one':worker(args.dataset,args.workflow,False);return
 import concurrent.futures as cf
 start=time.time();results=[]
 with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
  fs={pool.submit(worker,ds):ds for ds in sorted({r['dataset'] for r in registry()})}
  for future in cf.as_completed(fs):results.extend(future.result());put(W/'qa/PREDICTION_PROGRESS.json',dict(completed=len(results),elapsed_seconds=time.time()-start))
 assert len(results)==168 and all(g['configurations']==30 for g in results)
 put(W/'ALL5040_PREDICTIONS_LOCKED.json',dict(status='LOCKED_BEFORE_EVALUATION',maps=5040,pairs=168,configs=30,protocol_sha256=sha(W/'PROTOCOL_LOCK.json'),code_sha256=sha(Path(__file__)),gate_hashes={str(W/'predictions'/g['dataset']/g['workflow']/'PREDICTION_GATE.json'):sha(W/'predictions'/g['dataset']/g['workflow']/'PREDICTION_GATE.json') for g in results},seconds=time.time()-start))
if __name__=='__main__':main()
