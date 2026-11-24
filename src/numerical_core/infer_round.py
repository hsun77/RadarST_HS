from common import *
from object_evidence import *
import argparse,joblib
def infer_map(ds,wf,bundles,verifier):
 start=time.time();out=ROOT/'predictions'/ds/wf
 if (out/'PREDICTION_GATE.json').exists():return
 out.mkdir(parents=True,exist_ok=False);c=load_map(ds,wf);n=len(c['a']['spots']);base=c['a']['base'];members=c['members'];ids=c['ids'];genes=gene_evidence(c,ds)
 anchor=EX/'controls'/ds/wf/'CLEAN_V1';g=js(anchor/'PREDICTION_GATE.json');verifier.check(anchor/'PREDICTION_GATE.json')
 for name,h in g['files'].items():verifier.check(anchor/name,h)
 f=pd.read_parquet(anchor/'INSTANCE_REGISTRY.parquet');S0={c['lookup'][cid] for cid in f.candidate_id};expected=pd.read_csv(anchor/'UNIFIED_REFINEMENT_MAP.csv');assert expected.spot_id.tolist()==list(c['a']['spots']);assert np.array_equal(pd.factorize(expected.refined.astype(str))[0],pd.factorize(labels_from(S0,members,base).astype(str))[0])
 legal=[]
 for m in members:
  s=set(m);found={int(m[0])};stack=list(found)
  while stack:
   for j in c['adj'][stack.pop()]:
    if j in s and j not in found:found.add(j);stack.append(j)
  legal.append(found==s and 1<=len(s)<=10)
 legal=np.asarray(legal,bool);candidate_range=np.flatnonzero(legal).tolist();rows=[]
 for modelname,bundle in bundles.items():
  b1=modelname=='B1_ACTUAL_OBJECT';model=bundle['model'];cache={}
  for policy in ['VALUE','OBJECT_PRESERVE','DOMAIN_PARETO']:
   name=modelname+'__'+policy;S=S0.copy();seen=set();trace=[];lastpred=None
   for rnd in range(6):
    key=tuple(sorted(S))
    if key in seen:break
    seen.add(key)
    if key in cache:pred,footprints,conflicts=cache[key]
    else:
     ops=[int(i in S) for i in candidate_range];dynamic,footprints,conflicts=dynamic_features(c,S,candidate_range,ops,b1)
     X=c['x'].iloc[candidate_range].reset_index(drop=True).copy()
     for col in dynamic:X[col]=dynamic[col].to_numpy()
     if b1:
      for col in genes:X[col]=genes.iloc[candidate_range][col].to_numpy()
     arr=X[bundle['columns']].to_numpy(np.float32);assert np.isfinite(arr).all();pred=model.predict(arr)*bundle['scale'];pred[:,:6]=np.maximum(pred[:,:6],0)
     if not b1:
      owner=np.full(n,-1,int)
      for i in S:owner[members[i]]=i
      footprints=[];conflicts=[]
      for i in candidate_range:
       cf={i} if i in S else set(owner[members[i]])-{-1};foot=set(members[i])
       for j in cf:foot.update(members[j])
       footprints.append(foot);conflicts.append(cf)
     cache[key]=(pred,footprints,conflicts)
    net=pred[:,:3]-pred[:,3:6];score=net.sum(1);allowed=score>1e-12
    if policy=='OBJECT_PRESERVE':allowed&=(net[:,1]>=0)&(pred[:,7]>=0)
    if policy=='DOMAIN_PARETO':allowed&=(net>=0).all(1)&(pred[:,6:9].sum(1)>=0)&(pred[:,9]>=0)
    touched=set();changed=0
    for k in sorted(np.flatnonzero(allowed),key=lambda k:(-score[k],ids[candidate_range[k]])):
     i=candidate_range[k];foot=footprints[k];closed=foot|{j for s in foot for j in c['adj'][s]}
     if touched&closed:continue
     removal=i in S;cf=conflicts[k]
     # All frozen predictions in this batch refer to the same current state;
     # disjoint closed neighborhoods prevent directly interacting actions.
     if not cf<=S:continue
     S-=cf
     if not removal:S.add(i)
     touched|=closed;changed+=1;trace.append(dict(round=rnd,operation_remove=removal,candidate_id=ids[i],removed_ids=';'.join(ids[j] for j in sorted(cf)),predicted_net=float(score[k]),predicted_exact35=float(pred[k,7]),predicted_ARI=float(pred[k,9])))
    lastpred=pred
    if not changed:break
   labels=labels_from(S,members,base);used=set();selected=[]
   for i in sorted(S):
    m=set(members[i]);assert legal[i] and not m&used;used|=m;selected.append(c['frame'].iloc[i].to_dict())
   p=out/name;p.mkdir();pd.DataFrame(selected,columns=list(c['frame'])).to_parquet(p/'SELECTED_OBJECTS.parquet',index=False);pd.DataFrame(dict(spot_id=c['a']['spots'],refined=labels,edited=np.isin(np.arange(n),list(used)))).to_csv(p/'FINAL_LABELS.csv',index=False);pd.DataFrame(trace,columns=['round','operation_remove','candidate_id','removed_ids','predicted_net','predicted_exact35','predicted_ARI']).to_csv(p/'DECISIONS.csv',index=False)
   put(p/'SELECTION_GATE.json',dict(status='PASS',name=name,rounds=rnd+1,selected_count=len(S),actual_modified_spots=len(used),actual_modified_fraction=len(used)/n,composition=c['frame'].iloc[sorted(S)].groupby(['proposal_type','n_spots']).size().reset_index(name='count').to_dict('records'),reference_read=False,model_sha256=bundle['sha256'],training_datasets=bundle['training_datasets'],no_true_utility_guarantee=True,seconds=time.time()-start))
   rows.append(name);print('PREDICTED',ds,wf,name,len(S),time.time()-start,flush=True)
 put(out/'PREDICTION_GATE.json',dict(status='PASS',dataset=ds,workflow=wf,methods=rows,no_reference=True,source_hashes=verifier.checked,files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},utc=utc(),seconds=time.time()-start))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--workflow');a=ap.parse_args();guard('inference');v=Verify();bundles={}
 for name in ['B0_BASE_EVIDENCE','B1_ACTUAL_OBJECT']:
  p=ROOT/'models'/a.dataset/name;g=js(p/'MODEL_GATE.json');assert a.dataset not in g['training_datasets'] and g['strict_inner_state_exclusion'];assert sha(p/'MODEL.joblib')==g['model_sha256'];b=joblib.load(p/'MODEL.joblib');b['sha256']=g['model_sha256'];bundles[name]=b
 for r in pairs():
  if r['dataset']==a.dataset and (not a.workflow or r['workflow']==a.workflow):infer_map(a.dataset,r['workflow'],bundles,v)
if __name__=='__main__':main()
