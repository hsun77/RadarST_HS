import numpy as np,pandas as pd
def compute(c,z,rank):
 var=np.std(z,axis=0);keep=var>1e-7;z=np.asarray(z[:,keep],np.float32);rank=np.asarray(rank[:,keep],np.float32)
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

 return pd.DataFrame(rows)
