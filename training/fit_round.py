from common import *
from prepare_round import Y
import argparse,numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesRegressor
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outer',required=True);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();T=a.outer;guard('training',T)
 for gate in ['NESTED_ROUTE_AUDIT','INPUT_AND_ANCHOR_SOURCES_VERIFIED','SUPERVISION_VERIFIED','ACCESS_GUARD_TESTS','QUOTIENT_GEOMETRY_TESTS','Q2_TESTS']:assert js(ROOT/'audits'/(gate+'.json'))['status']=='PASS',gate
 lock=js(ROOT/'configs/IMPLEMENTATION_LOCK_PRE_FIT.json')
 for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,('PRE_FIT_CODE_CHANGED',rel)
 start=time.time();train=sorted({r['dataset'] for r in pairs()}-{T});parts=[];sources={}
 for J in train:
  for p,gname in [(ROOT/'targets'/J/'UPSTREAM.parquet','UPSTREAM_GATE.json'),(ROOT/'targets'/T/'nested'/J/'CONTEXT.parquet','CONTEXT_GATE.json')]:
   g=js(p.parent/gname);assert sha(p)==g['target_sha256'];t=pd.read_parquet(p);assert set(t.dataset)=={J}
   if gname=='CONTEXT_GATE.json':
    assert g['outer']==T and g['training_sample']==J
    for route in g['routes'].values():assert set(route['excluded'])=={T,J} and set(route['training_datasets']).isdisjoint({T,J})
   parts.append(t);sources[str(p)]=sha(p)
 table=pd.concat(parts,ignore_index=True);assert T not in set(table.dataset);y=table[Y].to_numpy(np.float32);good=np.isfinite(y).all(1);excluded=int((~good).sum());table=table.loc[good].reset_index(drop=True);y=y[good]
 # Do not let duplicate proposal identities multiply one physical intervention.
 physical={}
 for J in train:
  ident=pd.read_parquet(ROOT/'inputs'/J/'GraphST'/'CANDIDATES.parquet')
  physical.update({cid:hashlib.sha256(';'.join(sorted(m.split(';'))).encode()).hexdigest() for cid,m in zip(ident.candidate_id,ident.member_spots)})
 ph=table.candidate_id.map(physical);assert ph.notna().all();keys=table[['dataset','regime','workflow','operation_remove']].astype(str).agg('|'.join,axis=1)+'|'+ph;count=keys.value_counts();w=keys.map(lambda k:1/count[k]).to_numpy()
 # Equal section, then regime, then workflow: workflows are dependent repeats.
 for groupcols in [['dataset','regime','workflow'],['dataset','regime'],['dataset']]:
  k=table[groupcols].astype(str).agg('|'.join,axis=1);norm=pd.Series(w).groupby(k.to_numpy()).transform('sum').to_numpy();w/=norm
 w*=len(w)/w.sum();scale=np.sqrt(np.average((y-np.average(y,axis=0,weights=w))**2,axis=0,weights=w));scale[:9]=np.maximum(scale[:9],.01);scale[9]=max(scale[9],1e-8)
 base=[c for c in table if c.startswith(('raw::','interior_','guarded_','context_','struct_','molecule_','global_'))]+['operation_remove'];extra=[c for c in table if c.startswith(('obj_','res_'))]
 for name,cols in [('B0_BASE_EVIDENCE',base),('B1_ACTUAL_OBJECT',base+extra)]:
  out=ROOT/'models'/T/name;out.mkdir(parents=True,exist_ok=False);x=table[cols].to_numpy(np.float32);assert np.isfinite(x).all();assert not any('candidate_id' in c or c.startswith('y_') for c in cols)
  model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=8,max_features=.7,n_jobs=a.threads,random_state=20261104,bootstrap=False);model.fit(x,y/scale,sample_weight=w);model.n_jobs=1;joblib.dump(dict(model=model,columns=cols,targets=Y,scale=scale,heldout=T,training_datasets=train),out/'MODEL.joblib',compress=3)
  put(out/'MODEL_GATE.json',dict(status='PASS',outer=T,training_datasets=train,rows=len(table),invalid_target_rows_excluded=excluded,strict_inner_state_exclusion=True,no_candidate_dataset_or_workflow_ID_features=True,columns=cols,input_hashes=sources,model_sha256=sha(out/'MODEL.joblib'),seconds=time.time()-start))
  print('TRAINED',T,name,len(table),len(cols),time.time()-start,flush=True)
if __name__=='__main__':main()
