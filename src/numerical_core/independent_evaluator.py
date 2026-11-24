"""Independent evaluator: union-find, explicit set IoU, manual contingency metrics.
No RadarST evaluation / component / context imports. Frozen radius and label policies
are specifications reproduced here, not estimated from achieved performance.
"""
from common import *
import numpy as np, pandas as pd, math
from scipy.spatial import KDTree

COMPACT_BAD={'','nan','na','none','null','unknown','<na>','unassigned','uncertain','uncertain_overlap','undetermined'}
MACRO_BAD={'','nan','na','none','null','unknown','<na>','unassigned'}

def build_graph(xy):
    xy=np.asarray(xy,float);assert np.isfinite(xy).all()
    d=KDTree(xy).query(xy,k=2)[0][:,1] if len(xy)>1 else np.array([])
    d=d[np.isfinite(d)&(d>0)];radius=float(np.median(d)*1.25) if len(d) else 1.
    adj=[[] for _ in xy]
    for a,b in KDTree(xy).query_pairs(radius):adj[a].append(b);adj[b].append(a)
    return [sorted(v) for v in adj],radius

def regions(labels,adj):
    n=len(labels);assert n==len(adj);parent=list(range(n));rank=[0]*n
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,ns in enumerate(adj):
        for j in ns:
            if j<=i or labels[i]!=labels[j]:continue
            a,b=find(i),find(j)
            if a==b:continue
            if rank[a]<rank[b]:a,b=b,a
            parent[b]=a
            if rank[a]==rank[b]:rank[a]+=1
    groups={}
    for i in range(n):groups.setdefault(find(i),set()).add(i)
    return sorted(groups.values(),key=lambda s:min(s))

def reference_objects(truth,adj):
    return [s for s in regions(truth,adj) if str(truth[min(s)]).lower() not in COMPACT_BAD]

def quality(refs,preds,n):
    owner=np.empty(n,int)
    for j,s in enumerate(preds):owner[list(s)]=j
    rows=[]
    for t in refs:
        options=sorted({int(owner[i]) for i in t});best=max(options,key=lambda j:(len(t&preds[j])/len(t|preds[j]),-j))
        p=preds[best];inter=len(t&p)
        rows.append({'best_IoU':inter/len(t|p),'best_region_index':best,'best_region_size':len(p),'intersection':inter,'purity':inter/len(p),'dilution':1-inter/len(p),'coverage':inter/len(t),'split_region_count':len(options)})
    return rows

def information_metrics(truth,labels):
    mask=np.array([str(v).lower() not in MACRO_BAD for v in truth]);t=np.asarray(truth)[mask];p=np.asarray(labels)[mask]
    if len(t)==0:return {'ARI':float('nan'),'NMI':float('nan'),'macro_valid_spots':0,'macro_pred_labels':0,'macro_truth_labels':0}
    _,a=np.unique(t,return_inverse=True);_,b=np.unique(p,return_inverse=True);na=int(a.max())+1;nb=int(b.max())+1
    # Same registered degeneracy/NA policy as frozen baseline. Also expose standard ARI.
    c=np.bincount(a*nb+b,minlength=na*nb).reshape(na,nb).astype(float);r=c.sum(1);s=c.sum(0);n=c.sum()
    comb=lambda x:float(np.sum(x*(x-1)/2))
    ab,aa,bb=comb(c),comb(r),comb(s);expected=aa*bb/(n*(n-1)/2) if n>1 else 0.;den=(aa+bb)/2-expected
    ari=(ab-expected)/den if den else 1.
    ix,iy=np.nonzero(c);mi=float(np.sum(c[ix,iy]/n*np.log(c[ix,iy]*n/(r[ix]*s[iy]))));ht=-float(np.sum(r/n*np.log(r/n)));hp=-float(np.sum(s/n*np.log(s/n)));nmi=2*mi/(ht+hp) if ht+hp else 1.
    return {'ARI':ari if na>1 and nb>1 else float('nan'),'NMI':nmi if na>1 and nb>1 else float('nan'),'standard_ARI_including_degenerate':ari,'macro_valid_spots':int(n),'macro_pred_labels':nb,'macro_truth_labels':na}

def summarise(q,sizes,q0,qu):
    compact=(sizes>=1)&(sizes<=10);d=q-q0;du=q-qu
    v={'Mean_Best_IoU':float(q[compact].mean()),'exact':float((q[compact]>=1-1e-12).mean()),'near_0_8':float((q[compact]>=.8-1e-12).mean()),'delta_A0':float(d[compact].mean()),'gain_upstream':float(du[compact].mean()),'harm_A0':float(np.maximum(-d[compact],0).mean()),'improvement_A0':float(np.maximum(d[compact],0).mean()),'loss_fraction_A0':float((d[compact]<-1e-12).mean()),'loss_fraction_upstream':float((du[compact]<-1e-12).mean()),'new_exact_events':int(((q>=1-1e-12)&(q0<1-1e-12)&compact).sum()),'lost_exact_events':int(((q<1-1e-12)&(q0>=1-1e-12)&compact).sum()),'reference_compact_count':int(compact.sum()),'large_delta_A0':float(d[~compact].mean()) if (~compact).any() else float('nan')}
    for key,mask in [(f'size{k}',sizes==k) for k in range(1,11)]+[(f'size{lo}_{hi}',(sizes>=lo)&(sizes<=hi)) for lo,hi in [(1,2),(3,5),(6,10)]]:
        v[key+'_Mean']=float(q[mask].mean()) if mask.any() else float('nan');v[key+'_exact']=float((q[mask]>=1-1e-12).mean()) if mask.any() else float('nan')
    return v

