"""Official BANKSY H0/H1, auxiliary interface only; never passed into frozen heads."""
import numpy as np,pandas as pd
from anndata import AnnData
from banksy.main import generate_spatial_weights_fixed_nbrs
from banksy.embed_banksy import create_nbr_matrix

def compute_h0_h1(x,coords,radius):
    x=np.asarray(x,np.float32);weights={}
    for m in [0,1]:
        w,dist,theta=generate_spatial_weights_fixed_nbrs(np.asarray(coords,float),m=m,num_neighbours=6,decay_type='scaled_gaussian',max_radius=None,verbose=False)
        if np.any(np.diff(w.indptr)==0):raise ValueError('BANKSY_NO_PHYSICAL_NEIGHBOR')
        assert np.isfinite(w.data).all();weights[m]=w
    ad=AnnData(x);out=create_nbr_matrix(ad,{'scaled_gaussian':{'weights':weights}},'scaled_gaussian',max_m=1,center=True,variance_balance=False,verbose=False)
    # Independent exact formula, all rows; retain original official outputs.
    h0=weights[0]@x;h1=np.zeros(x.shape,float)
    w=weights[1]
    for i in range(len(x)):
        sl=slice(w.indptr[i],w.indptr[i+1]);ids=w.indices[sl]
        h1[i]=np.abs(w.data[sl]@(x[ids]-x[ids].mean(axis=0)))
    e0=float(np.max(np.abs(h0-out[0])));e1=float(np.max(np.abs(h1-out[1])));assert e0<1e-7 and e1<1e-7,(e0,e1)
    return out[0].astype(np.float32),out[1].astype(np.float32),weights,{'H0_formula_max_error':e0,'H1_centered_formula_max_error':e1,'official_API':'banksy.embed_banksy.create_nbr_matrix(center=True,variance_balance=False)','k':6,'m_orders':[0,1],'BANKSY_max_radius':None,'RadarST_connectivity_radius_unchanged':float(radius)}

def guarded_background(mem,adj,labels):
    own=list(labels[list(mem)]);dominant=max(set(own),key=lambda x:(own.count(x),str(x)))
    seen=set(mem);front=set(mem);rings=[]
    for _ in range(3):
        new={j for i in front for j in adj[i] if j not in seen};rings.append(new);seen|=new;front=new
    same={i for i in rings[1]|rings[2] if labels[i]==dominant}
    if len(same)>=5:return sorted(same)
    same={i for i in rings[0]|rings[1]|rings[2] if labels[i]==dominant}
    return sorted(same) if len(same)>=3 else sorted((rings[0]|rings[1]|rings[2])-mem)

def candidate_evidence(frame,spots,labels,adj,h0,h1):
    ix={s:i for i,s in enumerate(spots)};edge={}
    h1=h1/np.maximum(h1.std(axis=0),.2);h0=h0/np.maximum(h0.std(axis=0),.2)
    for i,neigh in enumerate(adj):
        for j in neigh:
            if i<j:edge[i,j]=float(np.sqrt(np.mean((h1[i]-h1[j])**2)))
    rows=[]
    for r in frame.itertuples(index=False):
        mem={ix[s] for s in r.member_spots.split(';')};bg=guarded_background(mem,adj,labels)
        contrast=float(np.sqrt(np.mean((h0[sorted(mem)].mean(axis=0)-h0[bg].mean(axis=0))**2))) if bg else 0.
        boundary=[];internal=[]
        for i in mem:
            for j in adj[i]:
                k=(min(i,j),max(i,j));v=edge[k]
                if j not in mem:boundary.append(v)
                elif i<j:internal.append(v)
        grad=(float(np.mean(boundary)) if boundary else 0.)-(float(np.mean(internal)) if internal else 0.)
        rows.append({'candidate_id':r.candidate_id,'proposal_type':r.proposal_type,'H0_guarded_contrast':contrast,'H1_boundary_minus_internal':grad,'guarded_background_count':len(bg)})
    f=pd.DataFrame(rows)
    for k in ['H0_guarded_contrast','H1_boundary_minus_internal']:f[k+'_rank']=f.groupby('proposal_type')[k].rank(method='average',pct=True)
    f['BANKSY_auxiliary_score']=.5*(f.H0_guarded_contrast_rank+f.H1_boundary_minus_internal_rank)
    return f
