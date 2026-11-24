"""Exact region-surgery engine; no model, feature definition or selection changes."""
from common import *
from object_evidence import labels_from,global_features,partition
from exact_engine import Counterfactual,maskof,indices as bitindices
import numpy as np,pandas as pd
def structure(c,cur,which):
 members=[c['members'][i] for i in which];n=len(cur);M=len(which);K=10;mem=np.full((M,K),n,np.int32)
 for row,m in enumerate(members):mem[row,:len(m)]=m
 sz=np.array([len(m) for m in members]);cc=partition(cur,c['adj']);sizes=np.bincount(cc);R=len(sizes);cp=np.r_[cc,R];sp=np.r_[sizes,0];g=np.sort(cp[mem],axis=1);valid=g<R;first=valid&np.c_[np.ones(M,bool),g[:,1:]!=g[:,:-1]];count=(g[:,:,None]==g[:,None,:]).sum(2)*first;gs=sp[g];best=np.where(first,count/np.maximum(sz[:,None]+gs-count,1),0).max(1);p=count/sz[:,None];ent=-np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)
 D=max(map(len,c['adj']),default=0);nei=np.full((n+1,D),n,np.int32)
 for i,ns in enumerate(c['adj']):nei[i,:len(ns)]=ns
 nb=nei[mem];ok=(mem[:,:,None]<n)&(nb<n);inside=(nb[:,:,:,None]==mem[:,None,None,:]).any(3);boundary=ok&~inside;internal=ok&inside&(mem[:,:,None]<nb);bcount=boundary.sum((1,2));icount=internal.sum((1,2));label=np.r_[cur,-1];diff=label[mem][:,:,None]!=label[nb]
 r={'struct_component_count':first.sum(1),'struct_best_existing_iou':best,'struct_existing_min_size':np.where(first,gs,n+1).min(1),'struct_existing_max_size':np.where(first,gs,0).max(1),'struct_existing_weighted_size':(gs*count).sum(1)/sz,'struct_complete_component_count':((count==gs)&first).sum(1),'struct_small_complete_count':((count==gs)&(gs<=10)&first).sum(1),'struct_small_partial_count':((count<gs)&(gs<=10)&first).sum(1),'struct_boundary_edge_count':bcount,'struct_internal_edge_count':icount,'struct_boundary_label_fraction':(boundary&diff).sum((1,2))/np.maximum(bcount,1),'struct_internal_label_fraction':(internal&diff).sum((1,2))/np.maximum(icount,1),'struct_label_entropy':ent}
 for lo,hi in [(1,2),(3,5),(6,10)]:r[f'struct_overlap_size{lo}_{hi}']=np.where((gs>=lo)&(gs<=hi),count,0).sum(1)
 return pd.DataFrame(r)
class FastTopology:
 def __init__(self,cur,base,adj,members,S):
  self.cur=cur;self.base=base;self.adj=adj;self.members=members;self.S=S;self.fresh=int(cur.max())+1;self.cc=partition(cur,adj);self.sizes=np.bincount(self.cc);self.owner=np.full(len(cur),-1,int)
  for i in S:self.owner[members[i]]=i
  self.bits=[maskof(np.flatnonzero(self.cc==k)) for k in range(len(self.sizes))];self.frag=object.__new__(Counterfactual);self.frag.base=cur;self.frag.adj=adj;self.frag.regions=self.bits;self.frag.cache={};self.region_label=[int(cur[next(bitindices(b))]) for b in self.bits]
 def action(self,i,remove=False):
  m=self.members[i];conf={i} if remove else set(self.owner[m]) - {-1};foot=set(map(int,m))
  for j in conf:foot.update(map(int,self.members[j]))
  restored={int(k) for j in conf for k in self.members[j]}
  if not remove:restored-=set(map(int,m))
  after={k:(int(self.base[k]) if k in restored or remove else self.fresh) for k in foot}
  # All changed members are restored first; candidate members receive a fresh ID.
  for j in conf:
   for k in self.members[j]:after[int(k)]=int(self.base[k])
  if not remove:
   for k in m:after[int(k)]=self.fresh
  region_ids=set(map(int,self.cc[list(foot)]))
  for k in restored:
   for j in self.adj[k]:
    if j not in foot and int(self.cur[j])==after[k]:region_ids.add(int(self.cc[j]))
  changed=maskof(foot);node_sizes=[];node_labels=[];fragments={};changed_node={};dsu=[]
  def newnode(size,label):
   k=len(node_sizes);node_sizes.append(size);node_labels.append(label);dsu.append(k);return k
  for ri in sorted(region_ids):
   entries=[]
   for b in self.frag.fragments(ri,self.bits[ri]&changed):entries.append((newnode(b.bit_count(),self.region_label[ri]),b))
   fragments[ri]=entries
  seen=set()
  for k in sorted(foot):
   if k in seen:continue
   lab=after[k];todo=[k];seen.add(k);part=[]
   while todo:
    v=todo.pop();part.append(v)
    for j in self.adj[v]:
     if j in foot and j not in seen and after[j]==lab:seen.add(j);todo.append(j)
   node=newnode(len(part),lab)
   for v in part:changed_node[v]=node
  def find(k):
   while dsu[k]!=k:dsu[k]=dsu[dsu[k]];k=dsu[k]
   return k
  for k in foot:
   for j in self.adj[k]:
    if j in foot or int(self.cur[j])!=after[k]:continue
    options=fragments[int(self.cc[j])];node=next(a for a,b in options if b&(1<<j));a,b=find(changed_node[k]),find(node)
    if a!=b:dsu[b]=a
  groups={}
  for k,size in enumerate(node_sizes):root=find(k);groups[root]=groups.get(root,0)+size
  sizes=np.array(list(groups.values()),np.int64);old=self.sizes[sorted(region_ids)];total=int(old.sum());assert sizes.sum()==total
  edited_delta=sum((after[k]!=int(self.base[k]))-(int(self.cur[k])!=int(self.base[k])) for k in foot)
  r={'res_operation_remove':int(remove),'res_conflict_objects':len(conf),'res_footprint_size':len(foot),'res_affected_spots':total,'res_before_regions':len(old),'res_after_regions':len(sizes),'res_region_delta':len(sizes)-len(old),'res_largest_fraction_delta':float(sizes.max()/total-old.max()/total),'res_edited_spot_delta':int(edited_delta)}
  for lo,hi in [(1,2),(3,5),(6,10),(11,10000000)]:
   a=old[(old>=lo)&(old<=hi)];b=sizes[(sizes>=lo)&(sizes<=hi)];r[f'res_count_delta_{lo}_{hi}']=len(b)-len(a);r[f'res_spot_delta_{lo}_{hi}']=int(b.sum()-a.sum())
  r['res_compact_partial_before']=int(np.sum((old<=10)&(old>len(m))));return r,foot,conf
def dynamic_features(c,S,which,operations=None,with_residual=True):
 cur=labels_from(S,c['members'],c['a']['base']);f=structure(c,cur,which)
 for k,val in global_features(c['a']['base'],cur).items():f[k]=val
 f['operation_remove']=np.zeros(len(which),int) if operations is None else np.asarray(operations,int);footprints=[];conflicts=[]
 if with_residual:
  top=FastTopology(cur,c['a']['base'],c['adj'],c['members'],S);rows=[]
  for i,op in zip(which,f.operation_remove):
   r,foot,conf=top.action(i,bool(op));rows.append(r);footprints.append(foot);conflicts.append(conf)
  f=pd.concat([f,pd.DataFrame(rows)],axis=1)
 return f,footprints,conflicts
