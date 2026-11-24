"""Observable candidate quality. Exact shell means follow CellCharter's graph
aggregation definition; our guarded candidate summaries are RadarST adaptations.
CellCharter BSD-3-Clause notice is retained in vendor/CellCharter/LICENSE.
"""
import numpy as np,pandas as pd
from scipy import sparse
from multiscale_cfar import adjacency_matrix

def shell_features(x,adj):
    a=adjacency_matrix(adj).astype(bool);seen=(a+sparse.eye(len(x),format='csr')).astype(bool)
    two=(a@a).astype(bool).astype(np.int8);two=two-two.multiply(seen);two.eliminate_zeros()
    def mean(m):
        n=np.asarray(m.sum(1)).ravel();return sparse.diags(1/np.maximum(n,1))@m@x
    return mean(a),mean(two)
def backgrounds(mem,adj,labels):
    dominant=max(set(labels[list(mem)]),key=lambda x:(np.count_nonzero(labels[list(mem)]==x),str(x)))
    seen=set(mem);front=set(mem);rings=[]
    for _ in range(3):
        nxt={j for i in front for j in adj[i] if j not in seen};rings.append(nxt);seen|=nxt;front=nxt
    bg={i for i in rings[1]|rings[2] if labels[i]==dominant}
    if len(bg)<3:bg={i for i in set.union(*rings) if labels[i]==dominant}
    if len(bg)<3:bg=set.union(*rings)
    return sorted(bg),sorted(rings[0])
def transform_candidates(frame,spots,labels,adj,pcs,multiscale,h1=None):
    ix={s:i for i,s in enumerate(spots)};x=np.asarray(pcs,float);h0,h2=shell_features(x,adj);rows=[]
    for r in frame.itertuples(index=False):
        mem={ix[s] for s in r.member_spots.split(';')};m=sorted(mem);bg,ring=backgrounds(mem,adj,np.asarray(labels,str));bg=bg or m;ring=ring or bg
        mi=x[m].mean(0);mb=x[bg].mean(0);variance=float(np.mean(np.var(x[m],axis=0)));bgvar=float(np.mean(np.var(x[bg],axis=0)))
        internal=[];boundary=[];cross=0
        for i in m:
            for j in adj[i]:
                distance=float(np.sqrt(np.mean((x[i]-x[j])**2)))
                if j not in mem:boundary.append(distance);cross+=int(labels[i]!=labels[j])
                elif i<j:internal.append(distance)
        row={'candidate_id':r.candidate_id,'interior_variance':variance,'interior_coherence':1/(1+variance),'interior_raw_norm':float(np.linalg.norm(mi)),'interior_internal_edge':float(np.mean(internal)) if internal else 0.,'interior_compactness':len(internal)/max(1,len(internal)+len(boundary)),
        'guarded_contrast':float(np.sqrt(np.mean((mi-mb)**2))),'guarded_background_variance':bgvar,'guarded_standardized_contrast':float(np.sqrt(np.mean((mi-mb)**2)))/np.sqrt(bgvar+1e-6),'guarded_background_count':len(bg),'guarded_boundary_minus_internal':(float(np.mean(boundary)) if boundary else 0.)-(float(np.mean(internal)) if internal else 0.),'guarded_boundary_cross_upstream_fraction':cross/max(1,len(boundary)),
        'context_hop1_contrast':float(np.sqrt(np.mean((h0[m].mean(0)-h0[bg].mean(0))**2))),'context_hop2_contrast':float(np.sqrt(np.mean((h2[m].mean(0)-h2[bg].mean(0))**2))),'context_hop1_variance':float(np.mean(np.var(h0[m],axis=0))),'context_hop2_variance':float(np.mean(np.var(h2[m],axis=0))),'context_scale_confidence':float(multiscale['scale_confidence'][m].mean()),'context_scale_choice_mean':float(multiscale['scale_choice'][m].mean())}
        for k in (1,2,4):
            rr=multiscale['residual'][k];row[f'context_residual{k}_contrast']=float(np.sqrt(np.mean((rr[m].mean(0)-rr[bg].mean(0))**2)))
        if h1 is not None:row['context_BANKSY_H1_boundary']=float(np.mean(h1[ring])-np.mean(h1[m]))
        rows.append(row)
    out=pd.DataFrame(rows);assert np.isfinite(out.drop(columns='candidate_id').to_numpy()).all();return out
