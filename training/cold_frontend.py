"""Reference-free expression + coordinates + one upstream map, freshly recomputed."""
from common import *
import argparse,types,numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--workflow',required=True);ap.add_argument('--run',default='cold_v2');a=ap.parse_args();guard('inference');start=time.time();CLEAN=OLD/'frozen_assets/clean_runtime';out=ROOT/'cold_runs'/a.run/a.dataset/a.workflow;out.mkdir(parents=True,exist_ok=False)
 samples=pd.read_csv(CLEAN/'manifests/SAMPLES.csv',keep_default_na=False);maps=pd.read_csv(CLEAN/'manifests/PAIRS.csv',keep_default_na=False);sample=next(r for r in samples.itertuples() if r.dataset==a.dataset);mapping=next(r for r in maps.itertuples() if r.dataset==a.dataset and r.method==a.workflow);coord=CLEAN/'data/coordinates'/(a.dataset+'.csv');manifest=js(CLEAN/'CLEAN_BASELINE_MODEL_MANIFEST.json');route=a.dataset if sample.cohort=='development' else 'FULL';md=next(r for r in manifest['models'] if r['heldout']==route);assert a.dataset not in md['training_samples'];models=CLEAN/md['directory']
 allowed={str(Path(p).resolve()).lower() for p in [sample.cfar_expression,sample.feature_expression,mapping.upstream,coord]}
 for name,h in md['files'].items():assert sha(models/name)==h;allowed.add(str((models/name).resolve()).lower())
 allowed.add(str((CLEAN/'configs/formal.json').resolve()).lower())
 # igraph probes its optional user configuration while importing official
 # BANKSY; this is a library config, not a scientific data or annotation input.
 igraph_config=Path.home()/'.igraphrc';allowed.add(str(igraph_config.resolve()).lower())
 def coldhook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   raw=os.fsdecode(args[0])
   if raw.lower()=='nul':return
   p=Path(raw).resolve();mode,flags=args[1:3];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if write:return
   # No historical state, candidates, accepted objects or feature arrays.
   forbidden=[ROOT/'inputs',ROOT/'features',OLD/'runs',AF,EX/'inputs',EX/'features',EX/'controls']
   if any(p.is_relative_to(q) for q in forbidden):raise RuntimeError('CACHED_STATE_IN_COLD_FRONTEND '+str(p))
   roots=[ROOT,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path('C:/Windows').resolve()]
   if not any(p.is_relative_to(q) for q in roots) and str(p).lower() not in allowed:raise RuntimeError('UNDECLARED_COLD_INPUT '+str(p))
 sys.addaudithook(coldhook);sys.path.insert(0,str(ROOT/'deploy_vendor'));sys.path.insert(0,str(ROOT/'deploy_modules'))
 from radarst import core
 from multiscale_cfar import transform
 from accelerated_context import transform as context_transform
 from BANKSY_FEATURE_MODULE import compute_h0_h1
 from gene_reader import read10x
 from gene_object_fresh_v2 import compute as gene_compute
 from object_evidence import assemble,labels_from
 from legacy_structure import structure_features
 coords=pd.read_csv(coord,dtype={'spot_id':str});ctx=core.make_context(sample.dataset,sample.sample,coords,sample.source)
 with threadpool_limits(limits=2,user_api='blas'),threadpool_limits(limits=1,user_api='openmp'):state=core.preprocess_coordinates(sample,CLEAN,coords)
 base=core.read_upstream(mapping.upstream,mapping.barcode_col,mapping.label_col,ctx['spot_ids']);pool=core.generate(sample,coords,state);frame=core.features(pool,base,ctx,state,a.workflow);bcode=pd.factorize(base,sort=False)[0];ix={s:i for i,s in enumerate(ctx['spot_ids'])};members=[np.array([ix[s] for s in m.split(';')],int) for m in frame.member_spots]
 h0,h1,weights,bcheck=compute_h0_h1(state['normalized_hvg'],ctx['coords'],core.functions().b2.infer_radius(ctx['coords']));ms=transform(state['pcs'],ctx['coords'],None,{'scales':[1,2,4],'heat_time':1.},ctx['adjacency']);old=core.functions().g.feature_frame(frame,sorted(frame.proposal_type.unique())).reset_index(drop=True).add_prefix('raw::');con=context_transform(frame,ctx['spot_ids'],base.to_numpy(),ctx['adjacency'],state['feature_pcs'],ms,h1).drop(columns='candidate_id');x=pd.concat([old,con.reset_index(drop=True),structure_features(frame,members,base.to_numpy(),ctx['adjacency'],state['feature_pcs'])],axis=1)
 edges=np.asarray([(i,int(j)) for i,ns in enumerate(ctx['adjacency']) for j in ns if i<j],np.int32).reshape(-1,2);data=dict(spots=np.asarray(ctx['spot_ids']),xy=ctx['coords'],edge=edges,base=bcode,pcs=state['feature_pcs']);f=frame[['candidate_id','proposal_type','member_spots','n_spots']];c=assemble(data,f,x)
 raw,genes,meta=read10x(Path(sample.cfar_expression),ctx['spot_ids'],normalize=False)
 if meta['integer_like_nonzero_fraction']>=1-1e-6:
  total=np.asarray(raw.sum(1)).ravel();raw=raw.multiply((1e4/np.maximum(total,1))[:,None]).tocsr();raw.data=np.log1p(raw.data)
 gx=raw.toarray().astype(np.float32);mean=gx.mean(0);std=gx.std(0);z=np.clip((gx-mean)/np.maximum(std,1e-4),-6,6);rank=np.empty_like(gx)
 for lo in range(0,gx.shape[1],512):rank[:,lo:lo+512]=((rankdata(gx[:,lo:lo+512],axis=0,method='average')-.5)/len(gx)*2-1).astype(np.float32)
 gene=gene_compute(c,z,rank);assert np.isfinite(x.to_numpy()).all() and np.isfinite(gene.to_numpy()).all();np.savez_compressed(out/'MAP.npz',**data);f.to_parquet(out/'CANDIDATES.parquet',index=False);x.to_parquet(out/'BASE_FEATURES.parquet',index=False);gene.to_parquet(out/'GENE_OBJECT_EVIDENCE.parquet',index=False)
 bundle=joblib.load(models/'selector_heads.joblib');ranker=joblib.load(models/'expected_gain.joblib');assert a.dataset not in bundle['training_datasets'];_,regions,alabel,_,_=core.infer(frame,base,ctx,bundle,ranker,js(CLEAN/'configs/formal.json'),a.workflow);S={c['lookup'][v] for v in regions.candidate_id};lab=labels_from(S,members,bcode);assert np.array_equal(pd.factorize(alabel.astype(str))[0],pd.factorize(lab.astype(str))[0]);regions.to_parquet(out/'FRESH_A0_SELECTED.parquet',index=False);pd.DataFrame(dict(spot_id=ctx['spot_ids'],refined=lab)).to_csv(out/'FRESH_A0_LABELS.csv',index=False)
 put(out/'COLD_FRONTEND_GATE.json',dict(status='PASS',dataset=a.dataset,workflow=a.workflow,candidates=len(f),no_reference_read=True,no_historical_candidates_scores_state_or_accepted_cache_read=True,model_route=route,models=md['files'],raw_input_hashes={str(p):sha(p) for p in [Path(sample.cfar_expression),Path(sample.feature_expression),Path(mapping.upstream),coord]},official_banksy_check=bcheck,gene_input_scale='integer-like:CP10k/log1p; otherwise provided values preserved',gene_count=len(genes),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},seconds=time.time()-start,utc=utc()));print('COLD_FRONTEND_DONE',a.dataset,a.workflow,len(f),time.time()-start)
if __name__=='__main__':main()
