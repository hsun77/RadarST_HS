"""RadarST adaptation, not an implementation of SpatialZoomer's full pipeline.
Keeps raw signal, evaluates short random-walk residuals and a sparse heat kernel.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import rankdata

def adjacency_matrix(adj):
    e=[(i,int(j)) for i,ns in enumerate(adj) for j in ns]
    a=sparse.csr_matrix((np.ones(len(e)),tuple(zip(*e))),shape=(len(adj),len(adj))) if e else sparse.csr_matrix((len(adj),len(adj)))
    a=a.maximum(a.T);a.setdiag(0);a.eliminate_zeros();return a
def rank(x):return rankdata(np.nan_to_num(np.asarray(x,float)),method='average')/max(1,len(x))
def transform(expression,coordinates,upstream_labels,config,adjacency):
    x=np.asarray(expression,float);a=adjacency_matrix(adjacency);degree=np.asarray(a.sum(1)).ravel()
    # A self loop only at isolated nodes makes constant signals invariant.
    a=a+sparse.diags((degree==0).astype(float));degree=np.asarray(a.sum(1)).ravel()
    p=sparse.diags(1/degree)@a;z=x.copy();smooth={};residual={}
    for k in range(1,5):
        z=p@z
        if k in config['scales']:smooth[k]=z.copy();residual[k]=x-z
    heat=expm_multiply(-(sparse.eye(len(x),format='csr')-p)*config['heat_time'],x,traceA=float(-np.sum(1-p.diagonal())*config['heat_time']))
    energy=np.stack([np.sqrt(np.mean(residual[k]**2,axis=1)) for k in config['scales']],axis=1)
    # Two disjoint PC-coordinate blocks give an observable agreement diagnostic,
    # NOT held-out genes or an independent biological replication.
    first=np.stack([np.sqrt(np.mean(residual[k][:,::2]**2,axis=1)) for k in config['scales']],axis=1)
    second=np.stack([np.sqrt(np.mean(residual[k][:,1::2]**2,axis=1)) if x.shape[1]>1 else first[:,j] for j,k in enumerate(config['scales'])],axis=1)
    stability=1-np.abs(first-second)/(first+second+1e-12)
    chosen=np.argmax(energy*stability,axis=1);adaptive=np.stack([residual[k] for k in config['scales']],axis=1)[np.arange(len(x)),chosen]
    return {'P':p,'raw':x,'smooth':smooth,'residual':residual,'heat':heat,'heat_residual':x-heat,'adaptive_residual':adaptive,'scale_choice':np.asarray(config['scales'])[chosen],'scale_confidence':stability[np.arange(len(x)),chosen],'residual_energy':energy,'oversmoothing_energy_ratio':np.asarray([np.var(smooth[k])/max(np.var(x),1e-12) for k in config['scales']])}
