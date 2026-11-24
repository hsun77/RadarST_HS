"""Exact offline counterfactual targets; no selector or model changes."""
import numpy as np,math,collections
from scipy.special import gammaln
from scipy.stats import hypergeom
EPS=1e-12
def maskof(ix):
 b=0
 for i in ix:b|=1<<int(i)
 return b
def indices(mask):
 while mask:
  low=mask&-mask;yield low.bit_length()-1;mask^=low
def partition(labels,adj,allowed=None):
 allowed=set(range(len(labels))) if allowed is None else set(allowed);seen=set();out=[]
 for start in sorted(allowed):
  if start in seen:continue
  todo=[start];seen.add(start);mask=0
  while todo:
   i=todo.pop();mask|=1<<i
   for j in adj[i]:
    if j in allowed and j not in seen and labels[j]==labels[i]:seen.add(j);todo.append(j)
  out.append(mask)
 return out
def iouvec(regions,truth):
 return np.asarray([max(((r&t).bit_count()/(r|t).bit_count() for r in regions if r&t),default=0.) for t in truth])
class Counterfactual:
 def __init__(self,base,adj,truth_masks):
  self.base=np.asarray(base);self.adj=adj;self.truth=truth_masks;self.regions=partition(base,adj);self.vertex_region={i:r for r,b in enumerate(self.regions) for i in indices(b)};self.cache={};self.new_label=int(np.max(base))+1
  self.quality=np.asarray([[(r&t).bit_count()/(r|t).bit_count() if r&t else 0. for r in self.regions] for t in truth_masks]);self.baseline=self.quality.max(axis=1);self.rank=np.argsort(-self.quality,axis=1,kind='stable');self.affected_truth=[np.flatnonzero(self.quality[:,r]>0) for r in range(len(self.regions))]
 def fragments(self,ri,removed):
  key=(ri,removed)
  if key in self.cache:return self.cache[key]
  remaining=self.regions[ri]&~removed
  if not remaining:result=[]
  else:
   boundary=set(j for i in indices(removed) for j in self.adj[i] if remaining&(1<<j))
   if len(boundary)<=1:result=[remaining]
   else:
    start=next(iter(boundary));seen={start};todo=[start];pending=boundary-{start};steps=0
    while todo and pending and steps<96:
     i=todo.pop();steps+=1
     for j in self.adj[i]:
      if j not in seen and remaining&(1<<j):seen.add(j);pending.discard(j);todo.append(j)
    if not pending:result=[remaining]
    else:result=partition(self.base,self.adj,indices(remaining))
  if len(self.cache)<80000:self.cache[key]=result
  return result
 def single(self,members,full_partition=False):
  # A real complete writeback map is created for every intervention.
  labels=self.base.copy();labels[members]=self.new_label;removed=maskof(members);touched={self.vertex_region[i] for i in members};new=[];fragmentation=0
  for ri in touched:
   ff=self.fragments(ri,self.regions[ri]&removed);new.extend(ff);fragmentation+=max(0,len(ff)-1)
  # A candidate can itself contain multiple spatially disconnected pieces.
  new.extend(partition(labels,self.adj,members));aff=set(j for ri in touched for j in self.affected_truth[ri]);values=self.baseline.copy()
  for j in aff:
   unchanged=next((self.quality[j,r] for r in self.rank[j] if r not in touched),0.);t=self.truth[j];added=max(((r&t).bit_count()/(r|t).bit_count() for r in new if r&t),default=0.);values[j]=max(unchanged,added)
  if full_partition:
   actual=partition(labels,self.adj);incremental=[r for i,r in enumerate(self.regions) if i not in touched]+new;assert set(actual)==set(incremental)
   assert np.allclose(values,iouvec(actual,self.truth),rtol=0,atol=1e-12)
  return labels,values,fragmentation
def ch2(x):return x*(x-1)/2
class Macro:
 """Exact ARI/NMI/AMI update for fresh-label candidate writeback, strict mask unchanged."""
 def __init__(self,truth,base,bad):
  self.valid=np.asarray([str(x).lower() not in bad for x in truth]);self.positions=np.flatnonzero(self.valid);self.t=np.full(len(truth),-1,int);self.p=np.asarray(base);self.t[self.valid]=np.unique(np.asarray(truth)[self.valid],return_inverse=True)[1];self.N=int(self.valid.sum());self.a=np.bincount(self.t[self.valid]);self.cols={}
  for pi in np.unique(self.p[self.valid]):self.cols[int(pi)]=np.bincount(self.t[self.valid&(self.p==pi)],minlength=len(self.a))
  self.b={p:int(v.sum()) for p,v in self.cols.items()};self.rowpairs=sum(ch2(x) for x in self.a);self.colpairs=sum(ch2(x) for x in self.b.values());self.samepairs=sum(ch2(x) for v in self.cols.values() for x in v);self.htrue=self.entropy(self.a);self.hpred=sum(self.hpiece(x) for x in self.b.values());self.mi_cols={p:self.colmi(v) for p,v in self.cols.items()};self.mi=sum(self.mi_cols.values());self.emicache={};self.emi=sum(self.emipiece(n) for n in self.b.values());self.baseline=self.calculate(self.samepairs,self.colpairs,self.mi,self.hpred,self.emi,len(self.cols))
 def hpiece(self,n):return -n/self.N*math.log(n/self.N) if n else 0.
 def entropy(self,counts):return sum(self.hpiece(int(n)) for n in counts)
 def colmi(self,v):
  b=int(v.sum())
  if not b:return 0.
  return sum(int(n)/self.N*math.log(int(n)*self.N/(int(self.a[i])*b)) for i,n in enumerate(v) if n)
 def emipiece(self,b):
  b=int(b)
  if b==0:return 0.
  if b in self.emicache:return self.emicache[b]
  value=0.
  for aa in self.a:
   a=int(aa);lo=max(1,a+b-self.N);hi=min(a,b)
   if lo>hi:continue
   k=np.arange(lo,hi+1,dtype=float);prob=hypergeom.pmf(k,self.N,a,b);value+=float(np.sum(prob*k/self.N*np.log(k*self.N/(a*b))))
  self.emicache[b]=value;return value
 def calculate(self,sp,cp,mi,hp,emi,ncols):
  if self.N==0 or len(self.a)<=1 or ncols<=1:return np.array([np.nan]*3)
  denom=ch2(self.N);expected=self.rowpairs*cp/denom if denom else 0;ari_d=(self.rowpairs+cp)/2-expected;ari=1. if ari_d==0 else (sp-expected)/ari_d;normal=(self.htrue+hp)/2;nmi=max(0.,mi)/normal if normal else 1.
  top=mi-emi;bottom=normal-emi;eps=np.finfo(float).eps;top=min(top,-eps) if top<0 else max(top,eps);bottom=min(bottom,-eps) if bottom<0 else max(bottom,eps);ami=top/bottom
  return np.asarray([ari,nmi,ami])
 def single(self,members):
  valid=[i for i in members if self.valid[i]]
  if not valid:return self.baseline.copy()
  removed=collections.defaultdict(lambda:np.zeros(len(self.a),int))
  for i in valid:removed[int(self.p[i])][self.t[i]]+=1
  sp=self.samepairs;cp=self.colpairs;mi=self.mi;hp=self.hpred;emi=self.emi;ncols=len(self.cols);new=np.zeros(len(self.a),int)
  for pi,v in removed.items():
   old=self.cols[pi];after=old-v;oldn=self.b[pi];n=int(after.sum());new+=v;sp+=sum(ch2(x) for x in after)-sum(ch2(x) for x in old);cp+=ch2(n)-ch2(oldn);mi+=self.colmi(after)-self.mi_cols[pi];hp+=self.hpiece(n)-self.hpiece(oldn);emi+=self.emipiece(n)-self.emipiece(oldn);ncols-=int(n==0)
  n=len(valid);sp+=sum(ch2(x) for x in new);cp+=ch2(n);mi+=self.colmi(new);hp+=self.hpiece(n);emi+=self.emipiece(n);ncols+=1
  return self.calculate(sp,cp,mi,hp,emi,ncols)
def delta_metrics(delta,masks):
 out={}
 for k,mask in masks.items():
  v=delta[mask];out.update({k+'_component_count':int(len(v)),k+'_net_sum':float(v.sum()),k+'_gain_mean':float(v.mean()) if len(v) else np.nan,k+'_improved_events':int((v>EPS).sum()),k+'_harm_events':int((v < -EPS).sum()),k+'_gain_magnitude':float(np.maximum(v,0).sum()),k+'_harm_magnitude':float(np.maximum(-v,0).sum()),k+'_max_harm':float(np.maximum(-v,0).max()) if len(v) else np.nan})
 return out
