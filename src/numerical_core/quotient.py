"""One unified label map with exactly preserved spatial instance partition."""
from common import *
from object_evidence import load_map
from independent_evaluator import regions
import numpy as np,pandas as pd
def reconstruct(cur,base,edges,adj,pcs,molecular=True,pair_optimize=False):
 rr=regions(cur,adj);n=len(cur);owner=np.empty(n,int)
 for k,s in enumerate(rr):owner[list(s)]=k
 nei=[set() for _ in rr]
 for a,b in edges:
  i,j=int(owner[a]),int(owner[b])
  if i!=j:nei[i].add(j);nei[j].add(i)
 scale=np.maximum(pcs.std(0),1e-8);x=pcs/scale;mass=np.array([len(s) for s in rr]);means=np.stack([x[list(s)].mean(0) for s in rr]);parent=[]
 for s in rr:
  values,cnt=np.unique(base[list(s)],return_counts=True);parent.append(int(values[np.argmax(cnt)]))
 colors=np.full(len(rr),-1,int);groupmass={};centers={}
 for i in sorted(range(len(rr)),key=lambda i:(-mass[i],min(rr[i]))):
  p=parent[i];used={colors[j] for j in nei[i] if colors[j]>=0 and parent[j]==p};available=[key for key in groupmass if key[0]==p and key[1] not in used]
  if available and molecular:
   key=min(available,key=lambda key:(float(mass[i]*groupmass[key]/(mass[i]+groupmass[key])*np.sum((means[i]-centers[key])**2)),key[1]))
  elif available:key=min(available,key=lambda key:key[1])
  else:
   col=0
   while (p,col) in groupmass:col+=1
   key=(p,col);groupmass[key]=0.;centers[key]=np.zeros(x.shape[1])
  old=groupmass[key];centers[key]=(centers[key]*old+means[i]*mass[i])/(old+mass[i]);groupmass[key]=old+mass[i];colors[i]=key[1]
 before_pair=sum(w*(w-1)/2 for w in groupmass.values());moves=0;sweeps=0
 if pair_optimize:
  while True:
   changed=0;sweeps+=1
   for i in sorted(range(len(rr)),key=lambda i:(-mass[i],min(rr[i]))):
    p=parent[i];old=(p,int(colors[i]));used={int(colors[j]) for j in nei[i] if parent[j]==p};allowed=[key for key in groupmass if key[0]==p and key!=old and key[1] not in used]
    if not allowed:continue
    best=max(allowed,key=lambda key:(groupmass[key],-key[1]));delta=int(mass[i])*(int(groupmass[best])-int(groupmass[old])+int(mass[i]))
    if delta<=0:continue
    groupmass[old]-=int(mass[i]);groupmass[best]+=int(mass[i]);colors[i]=best[1];moves+=1;changed+=1
   if not changed:break
 after_pair=sum(w*(w-1)/2 for w in groupmass.values());assert not pair_optimize or after_pair>=before_pair
 assert all(parent[i]!=parent[j] or colors[i]!=colors[j] for i,ns in enumerate(nei) for j in ns)
 labels=np.empty(n,int);code={};up_parent=np.empty(n,int)
 for i,s in enumerate(rr):
  key=(parent[i],int(colors[i]));lab=code.setdefault(key,len(code));labels[list(s)]=lab;up_parent[list(s)]=parent[i]
 actual=regions(labels,adj);assert {frozenset(s) for s in actual}=={frozenset(s) for s in rr},'MICROREGION_GEOMETRY_CHANGED'
 _,a=np.unique(base,return_inverse=True);_,b=np.unique(labels,return_inverse=True);nb=int(b.max())+1;joint=np.bincount(a*nb+b);ac=np.bincount(a);bc=np.bincount(b);choose=lambda z:float(np.sum(z*(z-1)/2));total=n*(n-1)/2;agree=choose(joint);disagreement=(choose(ac)+choose(bc)-2*agree)/total if total else 0.
 return labels,dict(connected_regions_before=len(rr),connected_regions_after=len(actual),output_domain_labels=len(code),input_domain_labels=len(np.unique(cur)),exact_partition_parity=True,parent_mismatch_fraction=float(np.mean(up_parent!=base)),upstream_pairwise_coassociation_disagreement=disagreement,colors_by_parent={str(p):len([key for key in code if key[0]==p]) for p in sorted(set(parent))},pair_objective_before=before_pair,pair_objective_after=after_pair,pair_optimizer_moves=moves,pair_optimizer_sweeps=sweeps,global_optimality_proven=False)
def main():
 guard('inference');base_lock=js(ROOT/'audits/ALL_PREDICTIONS_LOCKED.json');assert base_lock['status']=='PASS';v=Verify();allfiles={};count=0
 for reg in pairs():
  ds,wf=reg['dataset'],reg['workflow'];c=load_map(ds,wf);p=ROOT/'predictions'/ds/wf;g=js(p/'PREDICTION_GATE.json');items=[]
  for name in g['methods']:
   path=p/name/'FINAL_LABELS.csv';assert sha(path)==base_lock['files'][path.relative_to(ROOT).as_posix()];sg=js(path.parent/'SELECTION_GATE.json');items.append((name,path,sg['actual_modified_fraction']))
  for name in ['CLEAN_V1','C_ET']:
   p=EX/'controls'/ds/wf/name;cg=js(p/'PREDICTION_GATE.json');v.check(p/'PREDICTION_GATE.json');path=p/'UNIFIED_REFINEMENT_MAP.csv';v.check(path,cg['files'][path.name]);items.append((name,path,cg['modified_fraction']))
  for source_name,path,area in items:
   for mode in (['Q1','Q0_GEOMETRY_ONLY','Q2_PAIR_OPT'] if source_name in ['CLEAN_V1','C_ET'] else ['Q1','Q2_PAIR_OPT']):
    name=source_name+'__'+mode;out=ROOT/'quotient_predictions'/ds/wf/name;out.mkdir(parents=True,exist_ok=False);table=pd.read_csv(path);assert table.spot_id.tolist()==list(c['a']['spots']);lab,info=reconstruct(table.refined.to_numpy(),c['a']['base'],c['a']['edge'],c['adj'],c['a']['pcs'],molecular=mode=='Q1',pair_optimize=mode=='Q2_PAIR_OPT');pd.DataFrame(dict(spot_id=c['a']['spots'],refined=lab)).to_csv(out/'FINAL_LABELS.csv',index=False);put(out/'QUOTIENT_GATE.json',dict(status='PASS',method=name,no_reference=True,no_new_model=True,source_map_sha256=sha(path),actual_modified_fraction=area,area_definition='source selected-member writeback burden; quotient coassociation and parent effects reported independently',molecular_assignment=mode=='Q1',**info,files={'FINAL_LABELS.csv':sha(out/'FINAL_LABELS.csv')},utc=utc()));count+=1
  print('QUOTIENT_DONE',ds,wf,count,flush=True)
 files={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'quotient_predictions').rglob('*') if p.is_file()};put(ROOT/'audits/QUOTIENT_PREDICTIONS_LOCKED.json',dict(status='PASS',maps=count,files=files,all_connected_partitions_identical=True,source_hashes=v.checked,utc=utc()))
if __name__=='__main__':main()
