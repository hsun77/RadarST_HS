"""Deployment-observable evidence for actual object members and affected regions."""
from common import *
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from legacy_structure import partition,structure_features
def adjacency(n,edges):
 a=[[] for _ in range(n)]
 for i,j in edges:a[int(i)].append(int(j));a[int(j)].append(int(i))
 return [sorted(x) for x in a]
def labels_from(S,members,base):
 out=base.copy();offset=int(base.max())+1;seen=set()
 for i in sorted(S):
  m=set(members[i]);assert not m&seen;seen|=m;out[members[i]]=offset+i
 return out
def load_map(ds,wf,verifier=None):
 p=ROOT/'inputs'/ds/wf
 if (p/'INPUT_GATE.json').exists():
  g=js(p/'INPUT_GATE.json')
  with np.load(p/'MAP.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
  f=pd.read_parquet(p/'CANDIDATES.parquet');x=pd.read_parquet(p/'BASE_FEATURES.parquet');return assemble(a,f,x)
 v=verifier or Verify();src=EX/'inputs/maps'/ds/wf;ig=js(src/'INPUT_GATE.json');v.check(src/'INPUT_GATE.json')
 v.check(src/'OBSERVABLES.npz',ig['files']['OBSERVABLES.npz'])
 with np.load(src/'OBSERVABLES.npz',allow_pickle=False) as z:a={k:z[k] for k in ['spots','xy','edge','base']}
 fg=js(AF/'features'/ds/wf/'FEATURE_GATE.json');v.check(AF/'features'/ds/wf/'FEATURE_GATE.json')
 ip=AF/'features'/ds/wf/'CANDIDATE_IDENTITY.parquet';xp=AF/'features'/ds/wf/'OBSERVABLE_FEATURES.parquet';v.check(ip,fg['identity_sha256']);v.check(xp,fg['features_sha256']);f=pd.read_parquet(ip);x=pd.read_parquet(xp)
 front=OLD/'runs/a0_frontend'/ds;state=front/'state.npz'
 with np.load(state,allow_pickle=False) as z:assert list(z['spots'])==list(a['spots']);a['pcs']=z['feature_pcs'].astype(np.float32)
 # PCA is per-input unlabeled; no pooled fitting, pathway/ref labels or heads.
 p.mkdir(parents=True,exist_ok=False);np.savez_compressed(p/'MAP.npz',**a);f.to_parquet(p/'CANDIDATES.parquet',index=False);x.to_parquet(p/'BASE_FEATURES.parquet',index=False)
 put(p/'INPUT_GATE.json',dict(status='PASS',dataset=ds,workflow=wf,reference_read=False,candidates=len(f),files={q.name:sha(q) for q in p.iterdir() if q.is_file()},source_hashes={**v.checked,str(state):sha(state)}))
 return assemble(a,f,x)
def assemble(a,f,x):
 spots=list(a['spots']);ix={str(s):i for i,s in enumerate(spots)};members=[np.asarray([ix[s] for s in m.split(';')],np.int32) for m in f.member_spots]
 assert len(f)==len(x) and f.candidate_id.is_unique and len(set(spots))==len(spots)
 return dict(a=a,frame=f,x=x,members=members,adj=adjacency(len(spots),a['edge']),ids=f.candidate_id.tolist(),lookup={c:i for i,c in enumerate(f.candidate_id)})
def gene_evidence(c,ds,selected_indices=None):
 p=ROOT/'features'/ds/'GENE_OBJECT_EVIDENCE.parquet'
 if p.exists():
  t=pd.read_parquet(p);assert t.candidate_id.tolist()==c['ids'];return t.drop(columns='candidate_id')
 v=Verify();src=EX/'features/gene_inputs'/ds;g=js(src/'GENE_INPUT_GATE.json');v.check(src/'GENE_INPUT_GATE.json')
 for name in ['IDENTITY.npz','RELATIVE_EXPRESSION.npy','GENE_RANK.npy']:v.check(src/name,g['files'][name])
 with np.load(src/'IDENTITY.npz',allow_pickle=False) as z:assert z['spots'].tolist()==list(c['a']['spots'])
 z=np.load(src/'RELATIVE_EXPRESSION.npy',mmap_mode='r');rank=np.load(src/'GENE_RANK.npy',mmap_mode='r');var=np.std(z,axis=0);keep=var>1e-7;z=np.asarray(z[:,keep],np.float32);rank=np.asarray(rank[:,keep],np.float32)
 # Distributional features do not assume a shared PC/gene coordinate basis.
 rows=[];adj=c['adj']
 for m in c['members']:
  s=set(m);b=sorted({j for i in m for j in adj[i]}-s);b2=sorted({k for j in b for k in adj[j]}-s-set(b));r={};mu=z[m].mean(0);vv=z[m].var(0);rk=rank[m].mean(0)
  for tag,bg in [('ring1',b),('ring2',b2)]:
   if bg:
    zz=z[bg];bm=zz.mean(0);bv=zz.var(0);d=mu-bm;standard=d/np.sqrt((vv+bv)/2+.25);absolute=np.abs(standard);order=np.sort(absolute);top=min(32,len(order));positive=d>=0
    r.update({f'obj_{tag}_contrast_l2':float(np.sqrt(np.mean(d*d))),f'obj_{tag}_coherent_difference':float(np.mean(d*d/(1+vv))),f'obj_{tag}_gene_separation':float(np.mean(np.abs(rk-rank[bg].mean(0)))),f'obj_{tag}_top32_mean':float(order[-top:].mean()),f'obj_{tag}_positive_fraction':float(positive.mean()),f'obj_{tag}_dispersion_ratio':float(np.mean(vv)/(1+np.mean(bv))),f'obj_{tag}_background_count':len(bg)})
    for q,value in zip([25,50,75,90,95,99],np.percentile(absolute,[25,50,75,90,95,99])):r[f'obj_{tag}_standard_q{q}']=float(value)
   else:
    for key in ['contrast_l2','coherent_difference','gene_separation','top32_mean','positive_fraction','dispersion_ratio','background_count']+[f'standard_q{q}' for q in [25,50,75,90,95,99]]:r[f'obj_{tag}_{key}']=0.
  r['obj_internal_dispersion']=float(vv.mean());r['obj_internal_rank_dispersion']=float(rank[m].var(0).mean());r['obj_gene_feature_count_log']=float(np.log1p(z.shape[1]));r['obj_size']=len(m);rows.append(r)
 out=pd.DataFrame(rows);assert np.isfinite(out.to_numpy()).all();p.parent.mkdir(parents=True,exist_ok=True);pd.concat([pd.DataFrame({'candidate_id':c['ids']}),out],axis=1).to_parquet(p,index=False);put(p.parent/'GENE_EVIDENCE_GATE.json',dict(status='PASS',no_reference=True,source_hashes=v.checked,file_sha256=sha(p),columns=list(out),rows=len(out)))
 return out
def global_features(base,cur):
 _,cnt=np.unique(cur,return_counts=True);p=cnt/len(cur)
 return dict(global_log_spots=float(np.log1p(len(cur))),global_log_clusters=float(np.log1p(len(cnt))),global_largest_fraction=float(p.max()),global_entropy=float(-np.sum(p*np.log(p))),global_edited_fraction=float(np.mean(base!=cur)))
class LocalTopology:
 def __init__(self,cur,base,adj,members,S):
  self.cur=cur;self.base=base;self.adj=adj;self.members=members;self.S=S;self.cc=partition(cur,adj);self.sizes=np.bincount(self.cc);self.owner=np.full(len(cur),-1,int)
  for i in S:self.owner[members[i]]=i
  e=np.asarray([(i,j) for i,ns in enumerate(adj) for j in ns],np.int32).reshape(-1,2);self.graph=sparse.csr_matrix((np.ones(len(e),np.int8),(e[:,0],e[:,1])),shape=(len(cur),len(cur)))
  self.region_members=[np.flatnonzero(self.cc==k) for k in range(len(self.sizes))]
 def action(self,i,remove=False):
  m=self.members[i];conf={i} if remove else set(self.owner[m]) - {-1};foot=set(m)
  for j in conf:foot.update(self.members[j])
  region_ids=set(self.cc[list(foot)])
  restored={int(k) for j in conf for k in self.members[j]}
  if not remove:restored-=set(m)
  for k in restored:
   for j in self.adj[k]:
    if j not in foot and self.cur[j]==self.base[k]:region_ids.add(int(self.cc[j]))
  region=np.concatenate([self.region_members[k] for k in sorted(region_ids)]);after=self.cur[region].copy();lookup={int(j):k for k,j in enumerate(region)}
  for j in conf:
   for k in self.members[j]:after[lookup[int(k)]]=self.base[k]
  if not remove:
   for k in m:after[lookup[int(k)]]=int(self.cur.max())+1
  g=self.graph[region][:,region].tocoo();same=after[g.row]==after[g.col];n,cc=connected_components(sparse.csr_matrix((g.data[same],(g.row[same],g.col[same])),shape=g.shape),directed=False);sizes=np.bincount(cc);old=self.sizes[list(region_ids)];r={'res_operation_remove':int(remove),'res_conflict_objects':len(conf),'res_footprint_size':len(foot),'res_affected_spots':len(region),'res_before_regions':len(old),'res_after_regions':len(sizes),'res_region_delta':len(sizes)-len(old),'res_largest_fraction_delta':float(sizes.max()/len(region)-old.max()/len(region)),'res_edited_spot_delta':int(np.count_nonzero(after!=self.base[region])-np.count_nonzero(self.cur[region]!=self.base[region]))}
  for lo,hi in [(1,2),(3,5),(6,10),(11,10000000)]:
   a=old[(old>=lo)&(old<=hi)];b=sizes[(sizes>=lo)&(sizes<=hi)];r[f'res_count_delta_{lo}_{hi}']=len(b)-len(a);r[f'res_spot_delta_{lo}_{hi}']=int(b.sum()-a.sum())
  r['res_compact_partial_before']=int(np.sum((old<=10)&(old>len(m))));return r,foot,conf
def dynamic_features(c,S,indices,operations=None,with_residual=True):
 cur=labels_from(S,c['members'],c['a']['base']);f=structure_features(c['frame'].iloc[indices],[c['members'][i] for i in indices],c['a']['base'],c['adj'],c['a']['pcs'],cur,include_molecular=False);f.index=np.arange(len(f))
 for k,val in global_features(c['a']['base'],cur).items():f[k]=val
 f['operation_remove']=np.zeros(len(indices),int) if operations is None else np.asarray(operations,int)
 footprints=[];conflicts=[]
 if with_residual:
  top=LocalTopology(cur,c['a']['base'],c['adj'],c['members'],S);rows=[]
  for i,op in zip(indices,f.operation_remove):
   r,foot,conf=top.action(i,bool(op));rows.append(r);footprints.append(foot);conflicts.append(conf)
  f=pd.concat([f,pd.DataFrame(rows)],axis=1)
 return f,footprints,conflicts
