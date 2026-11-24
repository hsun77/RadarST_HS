"""Evaluate only the globally locked 5,040 predictions; no model fitting."""
import os,sys,json,time,hashlib,argparse,concurrent.futures as cf
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
W=Path('H:/RadarST_FigureRevision2_20261111');SRC=Path('H:/RadarST_2_ObjectValue_20261104');PAPER=Path('H:/RadarST_PROJECT/_manuscript_work_20261105')
sys.path.insert(0,str(SRC/'code'))
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from independent_evaluator import information_metrics
from object_metrics import ObjectEvaluator
from object_evidence import adjacency
from sensitivity import sha,put,registry,CONFIGS

def quality(labels,edge,refs,sizes):
 n=len(labels);same=labels[edge[:,0]]==labels[edge[:,1]];e=edge[same];g=csr_matrix((np.ones(len(e),np.int8),(e[:,0],e[:,1])),shape=(n,n));nc,cc=connected_components(g,directed=False);ps=np.bincount(cc,minlength=nc)
 owner=np.full(n,-1,np.int32)
 for i,s in enumerate(refs):owner[list(s)]=i
 mask=owner>=0;codes,count=np.unique(owner[mask]*nc+cc[mask],return_counts=True);ri=codes//nc;pi=codes%nc;ious=count/(sizes[ri]+ps[pi]-count)
 q=np.zeros(len(refs));np.maximum.at(q,ri,ious);best=(ious==q[ri]);anchors=np.full(nc,n);np.minimum.at(anchors,cc,np.arange(n));best_anchor=np.full(len(refs),n);np.minimum.at(best_anchor,ri[best],anchors[pi[best]]);chosen=best&(anchors[pi]==best_anchor[ri]);best_cc=np.zeros(len(refs),int);best_cc[ri[chosen]]=pi[chosen];inter=np.zeros(len(refs));inter[ri[chosen]]=count[chosen]
 coverage=inter/sizes;purity=inter/ps[best_cc];split=np.bincount(ri,minlength=len(refs));return q,coverage,purity,split

def one(r,tab,paper):
 ds,wf=r['dataset'],r['workflow'];inp=SRC/'inputs'/ds/wf;pred=W/'predictions'/ds/wf;out=W/'evaluation'/ds/wf;out.mkdir(parents=True,exist_ok=True)
 gatepath=out/'EVALUATION_GATE.json'
 if gatepath.exists():
  gate=json.loads(gatepath.read_text());assert sha(out/'MAP_RESULTS.csv')==gate['map_results_sha256'];assert sha(out/'COMPONENT_TRANSITIONS.csv')==gate['component_results_sha256'];return gate
 start=time.time();pg=json.loads((pred/'PREDICTION_GATE.json').read_text());assert pg['formal_E2_labels_exact_parity'] and pg['formal_initialization_exact_parity'];assert sha(pred/'FINAL_MAPS.npz')==pg['predictions_sha256']
 with np.load(inp/'MAP.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(pred/'FINAL_MAPS.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
 spots=a['spots'].astype(str);assert np.array_equal(spots,saved['spots']);assert saved['config_ids'].tolist()==[v['id'] for v in CONFIGS]
 truth=pd.read_csv(r['reference'],dtype=str,keep_default_na=False).set_index('spot_id').truth_label.reindex(spots);assert truth.notna().all()
 ev=ObjectEvaluator(truth.to_numpy(),adjacency(len(spots),a['edge']));mask=ev.sizes<=10;compact=np.flatnonzero(mask);lookup={s:i for i,s in enumerate(spots)}
 t=tab.loc[(tab.dataset_id==ds)&(tab.method==wf)&tab.component_size_spots.between(1,10)].copy().reset_index(drop=True)
 assert len(t)==len(compact)
 for row,j in zip(t.itertuples(),compact):assert {lookup[s] for s in row.reference_members.split(';')}==ev.refs[j]
 q0,_,_,_=quality(a['base'],a['edge'],ev.refs,ev.sizes);assert np.allclose(q0[mask],t.base_best_IoU,rtol=0,atol=1e-12)
 bmetrics=information_metrics(truth.to_numpy(),a['base']);old=paper[(paper.dataset==ds)&(paper.workflow==wf)];assert len(old)==1;old=old.iloc[0]
 pool=pd.read_parquet(inp/'CANDIDATES.parquet');members=[np.asarray([lookup[s] for s in v.split(';')],np.int32) for v in pool.member_spots]
 rows=[];details=[];vectors=[];maxparity=0.
 for ci,config in enumerate(CONFIGS):
  labels=saved['labels'][ci];S=saved['selected_indices'][saved['selected_offsets'][ci]:saved['selected_offsets'][ci+1]];expected=a['base'].copy();used=np.zeros(len(spots),bool)
  for j in S:
   assert not used[members[j]].any();used[members[j]]=True;expected[members[j]]=int(a['base'].max())+1+int(j)
  assert np.array_equal(labels,expected);q,cov,purity,split=quality(labels,a['edge'],ev.refs,ev.sizes);vectors.append(q);delta=q-q0;stats=information_metrics(truth.to_numpy(),labels)
  rec=dict(config=config['id'],expression_weight=config['expression_weight'],expected_gain_weight=config['expected_gain_weight'],retention_fraction=config['retention'],dataset=ds,workflow=wf,Mean_Best_IoU=float(q[mask].mean()),gain=float(delta[mask].mean()),exact=float((q[mask]>=1-1e-12).mean()),near_0_8=float((q[mask]>=.8-1e-12).mean()),loss_event_fraction=float((delta[mask]<-1e-12).mean()),harm_magnitude=float(np.maximum(-delta[mask],0).mean()),improvement_magnitude=float(np.maximum(delta[mask],0).mean()),compact_coverage=float(cov[mask].mean()),compact_purity=float(purity[mask].mean()),compact_fragment_count=float(split[mask].mean()),modified_spots=int((labels!=a['base']).sum()),candidate_member_spots=int(used.sum()),modified_fraction=float(used.mean()),accepted_candidates=len(S),large_mean=float(q[~mask].mean()) if (~mask).any() else np.nan,large_gain=float(delta[~mask].mean()) if (~mask).any() else np.nan,ARI=stats['ARI'],NMI=stats['NMI'],delta_ARI=stats['ARI']-bmetrics['ARI'],delta_NMI=stats['NMI']-bmetrics['NMI'])
  for lo,hi in [(1,2),(3,5),(6,10)]+[(k,k) for k in range(1,11)]:
   m=(ev.sizes>=lo)&(ev.sizes<=hi);name=f'size{lo}_{hi}' if lo!=hi else f'size{lo}';rec[name+'_Mean']=float(q[m].mean()) if m.any() else np.nan;rec[name+'_exact']=float((q[m]>=1-1e-12).mean()) if m.any() else np.nan
  # Compare the default on every map and every new configuration on the
  # initial engineering map against the independent explicit-set evaluator.
  if config['id']=='w80_r097' or (ds=='ST_PDAC_B' and wf=='GraphST'):
   independent,vec,quality_rows,*_=ev.evaluate(labels,True);error=float(np.abs(vec-q).max());maxparity=max(maxparity,error);assert error<1e-12
   assert np.allclose(cov,[v['coverage'] for v in quality_rows],atol=1e-12,rtol=0)
   assert np.allclose(purity,[v['purity'] for v in quality_rows],atol=1e-12,rtol=0)
   assert np.array_equal(split,[v['split_region_count'] for v in quality_rows])
   for key in ['Mean_Best_IoU','exact','near_0_8','ARI','NMI','size3_5_Mean']:
    assert np.isclose(rec[key],independent[key],atol=1e-12,rtol=0,equal_nan=True),(ds,wf,config['id'],key,rec[key],independent[key])
  if config['id']=='w80_r097':
   assert np.allclose(q[mask],t.after_best_IoU,rtol=0,atol=1e-12)
   for key in ['Mean_Best_IoU','exact','near_0_8','ARI','NMI','size3_5_Mean','modified_fraction','accepted_candidates']:
    assert np.isclose(rec[key],old[key],atol=1e-12,rtol=0,equal_nan=True),(ds,wf,key,rec[key],old[key])
   assert np.isclose(rec['gain'],old.delta_Mean_Best_IoU,atol=1e-12,rtol=0)
  d=t[['dataset_id','method','truth_component_id','truth_label','component_size_spots','reference_members','base_best_IoU']].copy();d.insert(0,'config',config['id']);d['after_best_IoU']=q[mask];d['delta_best_IoU']=delta[mask];d['coverage']=cov[mask];d['purity']=purity[mask];d['split_region_count']=split[mask];details.append(d);rows.append(rec)
 pd.DataFrame(rows).to_csv(out/'MAP_RESULTS.csv',index=False);pd.concat(details,ignore_index=True).to_csv(out/'COMPONENT_TRANSITIONS.csv',index=False);np.savez_compressed(out/'REFERENCE_VECTORS.npz',q=np.asarray(vectors),q0=q0,sizes=ev.sizes)
 gate=dict(status='PASS',dataset=ds,workflow=wf,configurations=30,independent_evaluator_max_error=maxparity,formal_default_frozen_endpoint_parity=True,reference_sha256=sha(r['reference']),prediction_sha256=pg['predictions_sha256'],map_results_sha256=sha(out/'MAP_RESULTS.csv'),component_results_sha256=sha(out/'COMPONENT_TRANSITIONS.csv'),seconds=time.time()-start);put(gatepath,gate);print('EVALUATED',ds,wf,round(gate['seconds'],2),flush=True);return gate

def worker(ds):
 tab=pd.read_csv(PAPER/'PAPER_ALL_COMPONENT_EVENTS.csv');paper=pd.read_csv(PAPER/'PAPER_ALL322_MAP_METRICS.csv');return [one(r,tab,paper) for r in registry() if r['dataset']==ds]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();lock=W/'ALL5040_PREDICTIONS_LOCKED.json';assert lock.exists(),'Never evaluate before all 5,040 predictions are locked';locked=json.loads(lock.read_text());assert locked['maps']==5040
 for p,h in locked['gate_hashes'].items():
  assert sha(p)==h;g=json.loads(Path(p).read_text());assert sha(Path(p).parent/'FINAL_MAPS.npz')==g['predictions_sha256']
 start=time.time();gates=[]
 with cf.ProcessPoolExecutor(max_workers=args.workers) as exe:
  fs=[exe.submit(worker,ds) for ds in sorted({r['dataset'] for r in registry()})]
  for f in cf.as_completed(fs):gates.extend(f.result())
 assert len(gates)==168
 maps=pd.concat([pd.read_csv(W/'evaluation'/r['dataset']/r['workflow']/'MAP_RESULTS.csv') for r in registry()],ignore_index=True);assert len(maps)==5040;maps.to_csv(W/'evaluation/ALL5040_MAP_RESULTS.csv',index=False)
 sec=maps.groupby(['config','dataset']).mean(numeric_only=True).reset_index();sec.to_csv(W/'evaluation/ALL360_SECTION_RESULTS.csv',index=False)
 wf=maps.groupby(['config','workflow']).mean(numeric_only=True).reset_index();wf.to_csv(W/'evaluation/ALL420_WORKFLOW_RESULTS.csv',index=False)
 summary=[]
 for config in CONFIGS:
  s=sec[sec.config.eq(config['id'])];f=wf[wf.config.eq(config['id'])];m=maps[maps.config.eq(config['id'])];assert len(s)==12 and len(f)==14 and len(m)==168
  rec=dict(config=config['id'],expression_weight=config['expression_weight'],expected_gain_weight=config['expected_gain_weight'],retention_fraction=config['retention'],formal_default=config['id']=='w80_r097',sections=12,workflows=14,maps=168,sample_equal_size_1_10_gain=float(s.gain.mean()),minimum_method_gain=float(f.gain.min()),minimum_sample_gain=float(s.gain.min()),positive_methods=int((f.gain>0).sum()),positive_samples=int((s.gain>0).sum()),median_delta_ARI=float(m.delta_ARI.median()))
  for key in ['Mean_Best_IoU','exact','near_0_8','loss_event_fraction','harm_magnitude','improvement_magnitude','modified_fraction','accepted_candidates','size1_2_Mean','size1_2_exact','size3_5_Mean','size3_5_exact','size6_10_Mean','size6_10_exact','ARI','NMI','delta_ARI','delta_NMI']:rec[key]=float(s[key].mean())
  summary.append(rec)
 summary=pd.DataFrame(summary);summary.to_csv(W/'evaluation/PARAMETER_SENSITIVITY_30.csv',index=False);(W/'evaluation/PARAMETER_SENSITIVITY_30.json').write_text(json.dumps(summary.to_dict('records'),indent=2,allow_nan=False),encoding='utf-8')
 components=pd.concat([pd.read_csv(W/'evaluation'/r['dataset']/r['workflow']/'COMPONENT_TRANSITIONS.csv') for r in registry()],ignore_index=True);components.to_csv(W/'evaluation/ALL_COMPONENT_TRANSITIONS.csv',index=False)
 put(W/'evaluation/EVALUATION_COMPLETE.json',dict(status='PASS',maps=len(maps),configurations=30,sections=12,workflows=14,component_events=len(components),max_independent_error=max(g['independent_evaluator_max_error'] for g in gates),default_all168_labels_and_endpoints_exact_parity=True,prediction_lock_sha256=sha(lock),code_sha256=sha(Path(__file__)),summary_sha256=sha(W/'evaluation/PARAMETER_SENSITIVITY_30.csv'),seconds=time.time()-start,evaluation_gates={str(W/'evaluation'/g['dataset']/g['workflow']/'EVALUATION_GATE.json'):sha(W/'evaluation'/g['dataset']/g['workflow']/'EVALUATION_GATE.json') for g in gates}));print(summary[['config','sample_equal_size_1_10_gain','minimum_method_gain','positive_samples','positive_methods']].to_string(index=False),flush=True)
if __name__=='__main__':main()
