"""Algebraically equivalent batch candidate moments; no new feature/target."""
import numpy as np,pandas as pd
from scipy import sparse
from context_features import backgrounds,shell_features

def transform(frame,spots,labels,adj,pcs,ms,h1):
    ix={s:i for i,s in enumerate(spots)};x=np.asarray(pcs,float);labels=np.asarray(labels,str);h0,h2=shell_features(x,adj);mem=[];bg=[];rings=[]
    for ss in frame.member_spots:
        m={ix[s] for s in ss.split(';')};b,r=backgrounds(m,adj,labels);b=b or sorted(m);r=r or b;mem.append(sorted(m));bg.append(b);rings.append(r)
    def matrix(rows):
        lengths=np.asarray([len(r) for r in rows]);return sparse.csr_matrix((np.repeat(1/lengths,lengths),(np.repeat(np.arange(len(rows)),lengths),np.concatenate(rows))),shape=(len(rows),len(spots)))
    M=matrix(mem);B=matrix(bg);G=matrix(rings);mi=M@x;mb=B@x
    variance=np.maximum((M@(x*x)-mi*mi).mean(1),0);bgvar=np.maximum((B@(x*x)-mb*mb).mean(1),0);contrast=np.sqrt(np.mean((mi-mb)**2,axis=1))
    edges={(i,j):float(np.sqrt(np.mean((x[i]-x[j])**2))) for i,ns in enumerate(adj) for j in ns if i<j};inside=[];boundary=[];compact=[];cross=[]
    for m in mem:
        s=set(m);a=[];b=[];cc=0
        for i in m:
            for j in adj[i]:
                v=edges[min(i,j),max(i,j)]
                if j not in s:b.append(v);cc+=int(labels[i]!=labels[j])
                elif i<j:a.append(v)
        inside.append(float(np.mean(a)) if a else 0.);boundary.append(float(np.mean(b)) if b else 0.);compact.append(len(a)/max(1,len(a)+len(b)));cross.append(cc/max(1,len(b)))
    d={'candidate_id':frame.candidate_id.to_numpy(),'interior_variance':variance,'interior_coherence':1/(1+variance),'interior_raw_norm':np.linalg.norm(mi,axis=1),'interior_internal_edge':inside,'interior_compactness':compact,'guarded_contrast':contrast,'guarded_background_variance':bgvar,'guarded_standardized_contrast':contrast/np.sqrt(bgvar+1e-6),'guarded_background_count':[len(b) for b in bg],'guarded_boundary_minus_internal':np.asarray(boundary)-inside,'guarded_boundary_cross_upstream_fraction':cross,'context_hop1_contrast':np.sqrt(np.mean((M@h0-B@h0)**2,axis=1)),'context_hop2_contrast':np.sqrt(np.mean((M@h2-B@h2)**2,axis=1)),'context_hop1_variance':np.maximum((M@(h0*h0)-(M@h0)**2).mean(1),0),'context_hop2_variance':np.maximum((M@(h2*h2)-(M@h2)**2).mean(1),0),'context_scale_confidence':M@ms['scale_confidence'],'context_scale_choice_mean':M@ms['scale_choice']}
    for k in [1,2,4]:d[f'context_residual{k}_contrast']=np.sqrt(np.mean((M@ms['residual'][k]-B@ms['residual'][k])**2,axis=1))
    # Preserve the official cached array's float32 reduction semantics exactly.
    # Averaging rows then float64 sparse weighting changes the old reduction by ~1e-7.
    d['context_BANKSY_H1_boundary']=[float(np.mean(h1[r])-np.mean(h1[m])) for r,m in zip(rings,mem)]
    return pd.DataFrame(d)
