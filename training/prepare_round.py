from common import *
from object_evidence import *
import argparse
Y=[f'y_{kind}_{lo}_{hi}' for kind in ['gain','harm','exact'] for lo,hi in [(1,2),(3,5),(6,10)]]+['y_ARI_delta']
def append_features(t,c,ds,S):
 indices=[c['lookup'][cid] for cid in t.candidate_id];ops=t.operation_remove.to_numpy() if 'operation_remove' in t else np.zeros(len(t),int)
 # Match frozen current-state structure before adding genuinely new features.
 dyn,_,_=dynamic_features(c,S,indices,ops,True)
 shared=[k for k in dyn if k in t and k!='operation_remove'];assert np.allclose(t[shared].to_numpy(float),dyn[shared].to_numpy(float),rtol=0,atol=1e-6),('OLD_STATE_FEATURE_PARITY',ds,shared)
 genes=gene_evidence(c,ds).iloc[indices].reset_index(drop=True)
 out=t.reset_index(drop=True).copy()
 for k in dyn:out[k]=dyn[k].to_numpy()
 for k in genes:out[k]=genes[k].to_numpy()
 return out
def upstream(ds,v):
 out=ROOT/'targets'/ds/'UPSTREAM.parquet'
 if out.exists():
  g=js(out.parent/'UPSTREAM_GATE.json');assert sha(out)==g['target_sha256'];return out
 tg=js(AF/'targets'/ds/'TARGET_V2_GATE.json');mg=js(AF/'targets'/ds/'MACRO_TARGET_GATE.json');v.check(AF/'targets'/ds/'TARGET_V2_GATE.json');v.check(AF/'targets'/ds/'MACRO_TARGET_GATE.json');p=AF/'targets'/ds/'WRITEBACK_TARGETS_V2.parquet';q=AF/'targets'/ds/'MACRO_SUPERVISION.parquet';v.check(p,tg['target_sha256']);v.check(q,mg['target_sha256'])
 table=pd.read_parquet(p);table=table.loc[table.context.eq(0)].copy();macro=pd.read_parquet(q);macro=macro.loc[macro.context.eq(0)];table=table.merge(macro,on=['dataset','workflow','candidate_id','context'],validate='one_to_one');table['operation_remove']=0;table['regime']='upstream';parts=[]
 for wf,t in table.groupby('workflow',sort=True):
  c=load_map(ds,wf,v);parts.append(append_features(t,c,ds,set()));print('UPSTREAM_AUGMENTED',ds,wf,len(t),flush=True)
 out.parent.mkdir(parents=True,exist_ok=True);pd.concat(parts,ignore_index=True).to_parquet(out,index=False);put(out.parent/'UPSTREAM_GATE.json',dict(status='PASS',dataset=ds,target_sha256=sha(out),source_files=v.checked,no_supervised_generator=True));return out
def nested(T,J,v):
 out=ROOT/'targets'/T/'nested'/J/'CONTEXT.parquet'
 if out.exists():
  g=js(out.parent/'CONTEXT_GATE.json');assert sha(out)==g['target_sha256'];return
 root=AF/'onpolicy_targets'/T/J;gatepath=root/'TARGET_GATE.json';g=js(gatepath);v.check(gatepath);assert g['outer_heldout']==T and g['training_section']==J
 p=root/'TRAINING_TABLE.parquet';v.check(p,g['target_sha256']);tab=pd.read_parquet(p);parts=[];routes={}
 for wf,t in tab.groupby('workflow',sort=True):
  key='__'.join(sorted([T,J]));state=AF/'onpolicy_states'/key/J/wf;sgp=state/'STATE_GATE.json';sg=js(sgp);v.check(sgp,g['state_inputs'][wf]['state_gate_sha256']);assert set(sg['excluded'])=={T,J} and set(sg['training_datasets']).isdisjoint({T,J}) and len(sg['training_datasets'])==10
  for name,h in sg['files'].items():v.check(state/name,h)
  c=load_map(J,wf,v);selected=pd.read_csv(state/'SELECTED_OBJECTS.csv');S={c['lookup'][cid] for cid in selected.candidate_id};labels=pd.read_csv(state/'CURRENT_LABELS.csv');assert 'current_code' in labels and labels.spot_id.tolist()==list(c['a']['spots']);assert np.array_equal(pd.factorize(labels.current_code.astype(str))[0],pd.factorize(labels_from(S,c['members'],c['a']['base']).astype(str))[0]),('STATE_LABEL_PARITY',state,list(labels))
  parts.append(append_features(t,c,J,S));routes[wf]=dict(excluded=sg['excluded'],training_datasets=sg['training_datasets'],state_gate_sha256=sha(sgp));print('NESTED_AUGMENTED',T,J,wf,len(t),flush=True)
 out.parent.mkdir(parents=True,exist_ok=True);tt=pd.concat(parts,ignore_index=True);tt.to_parquet(out,index=False);put(out.parent/'CONTEXT_GATE.json',dict(status='PASS',outer=T,training_sample=J,rows=len(tt),routes=routes,target_sha256=sha(out),source_files=v.checked,utc=utc()))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--upstream');ap.add_argument('--outer');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();guard('target' if a.upstream else 'training',a.outer);v=Verify();start=time.time()
 if a.upstream:upstream(a.upstream,v)
 else:
  train=sorted({r['dataset'] for r in pairs()}-{a.outer})
  if a.smoke:train=['ST_PDAC_A'] if a.outer!='ST_PDAC_A' else ['ST_PDAC_B']
  for J in train:nested(a.outer,J,v)
 print('PREPARED',a.upstream or a.outer,time.time()-start,flush=True)
if __name__=='__main__':main()
