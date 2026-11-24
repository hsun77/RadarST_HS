import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
def partition(labels,adj):
    edges=np.asarray([(i,j) for i,ns in enumerate(adj) for j in ns],int).reshape(-1,2)
    keep=labels[edges[:,0]]==labels[edges[:,1]];e=edges[keep]
    return connected_components(sparse.csr_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(labels),len(labels))),directed=False)[1]
def structure_features(frame,members,base,adj,pcs,current=None,include_molecular=True):
    base=np.asarray(base);cur=base if current is None else np.asarray(current);cc=partition(cur,adj);sizes=np.bincount(cc);rows=[]
    for r,m in zip(frame.itertuples(),members):
        s=set(m);ids,counts=np.unique(cc[m],return_counts=True);best=counts/(len(m)+sizes[ids]-counts);bound=[(i,j) for i in m for j in adj[i] if j not in s];internal=[(i,j) for i in m for j in adj[i] if j in s and i<j]
        row={'struct_component_count':len(ids),'struct_best_existing_iou':float(best.max()),'struct_existing_min_size':int(sizes[ids].min()),'struct_existing_max_size':int(sizes[ids].max()),'struct_existing_weighted_size':float(np.dot(sizes[ids],counts)/len(m)),'struct_complete_component_count':int(np.sum(counts==sizes[ids])),'struct_small_complete_count':int(np.sum((counts==sizes[ids])&(sizes[ids]<=10))),'struct_small_partial_count':int(np.sum((counts<sizes[ids])&(sizes[ids]<=10))),'struct_boundary_edge_count':len(bound),'struct_internal_edge_count':len(internal),'struct_boundary_label_fraction':float(np.mean([cur[i]!=cur[j] for i,j in bound])) if bound else 0.,'struct_internal_label_fraction':float(np.mean([cur[i]!=cur[j] for i,j in internal])) if internal else 0.,'struct_label_entropy':float(-np.sum(counts/len(m)*np.log(counts/len(m))))}
        for lo,hi in [(1,2),(3,5),(6,10)]:row[f'struct_overlap_size{lo}_{hi}']=int(np.sum(counts[(sizes[ids]>=lo)&(sizes[ids]<=hi)]))
        if not include_molecular:rows.append(row);continue
        x=pcs[m];bg=sorted({j for i,j in bound});d=np.linalg.norm(x-x.mean(0),axis=1);row['molecule_radius_max']=float(d.max());row['molecule_radius_q75']=float(np.quantile(d,.75));row['molecule_pairwise_max']=float(np.max(np.linalg.norm(x[:,None]-x[None,:],axis=2)))
        if bg:
            dif=pcs[bg]-x.mean(0);dist=np.linalg.norm(dif,axis=1);row['molecule_background_min']=float(dist.min());row['molecule_background_q25']=float(np.quantile(dist,.25));row['molecule_boundary_separation']=row['molecule_background_q25']/(1+row['molecule_radius_max'])
        else:row.update({k:0. for k in ['molecule_background_min','molecule_background_q25','molecule_boundary_separation']})
        rows.append(row)
    return pd.DataFrame(rows)
