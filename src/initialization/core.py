from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib,json,math
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score,silhouette_score

def functions():
    names=['b67','b2','b71','helper','i','d','t','u','d22','e','signed','targets','a','g','k','r','sel','c','m','internal','ext','expr']
    f=SimpleNamespace(**{n:importlib.import_module('radarst.legacy.'+n) for n in names})
    for m in (f.b71,f.helper,f.i,f.t,f.d22,f.e,f.targets,f.r):m.B67=f.b67
    f.helper.B71=f.b71;f.t.D=f.d;f.u.T=f.t;f.d22.T=f.t;f.d22.U=f.u;f.e.D22=f.d22;f.e.T=f.t
    f.signed.D=f.d;f.a.D=f.d;f.g.A=f.a;f.k.G=f.g;f.r.E=f.e;f.r.T=f.t;f.sel.R=f.r
    return f

def candidate_identity(dataset,typ,members):
    obj=[str(dataset),str(typ),sorted(set(map(str,members)))]
    text=json.dumps(obj,ensure_ascii=True,separators=(',',':'))
    return hashlib.sha256(text.encode('ascii')).hexdigest()

def strict_join(left,right,on):
    if left[on].duplicated().any() or right[on].duplicated().any():raise ValueError('Non-unique candidate join')
    result=left.merge(right,on=on,how='left',validate='one_to_one',indicator=True)
    if len(result)!=len(left) or not result['_merge'].eq('both').all():raise ValueError('Candidate coverage mismatch')
    return result.drop(columns='_merge')

def read_expression(path,kind):
    data=sc.read_10x_h5(str(path)) if kind=='10x_h5' else ad.read_h5ad(path)
    data.var_names_make_unique();data.obs_names=data.obs_names.astype(str)
    if not data.obs_names.is_unique:raise ValueError('Duplicate expression barcodes')
    return data

def feature_pcs(path,kind,spots):
    data=read_expression(path,kind)
    if set(spots)-set(data.obs_names):raise ValueError('Expression alignment missing')
    data=data[spots].copy();sc.pp.normalize_total(data,target_sum=1e4);sc.pp.log1p(data)
    if data.n_vars>3000:sc.pp.highly_variable_genes(data,n_top_genes=3000,flavor='seurat',subset=True)
    sc.pp.pca(data,n_comps=max(2,min(30,data.n_obs-1,data.n_vars-1)),zero_center=True,random_state=20260805)
    x=np.asarray(data.obsm['X_pca'],dtype=np.float32);med=np.median(x,axis=0)
    return ((x-med)/np.maximum(1.4826*np.median(np.abs(x-med),axis=0),1e-3)).astype(np.float32)

def read_upstream(path,bc,lc,spots):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if bc not in df or lc not in df:raise ValueError('Declared upstream columns missing')
    if df[bc].duplicated().any():raise ValueError('Duplicate upstream barcodes')
    series=df.set_index(bc)[lc].reindex(spots)
    fraction=float(series.notna().mean())
    if fraction<.5:raise ValueError(f'Unverified upstream alignment: {int(series.isna().sum())} missing')
    # Preserve the formal missing-prediction sentinel, never invent a spot label
    # or apply the historical positional fallback.
    return series.fillna('missing_pred').replace('','missing_pred').astype(str)

def make_context(dataset,sample,coords,source):
    f=functions();spots=coords.spot_id.astype(str).tolist();xy=coords[['x','y']].to_numpy(float)
    graph=f.ext.radius_adjacency(xy) if source=='external_second' else f.b2.adjacency_at_radius(xy,f.b2.infer_radius(xy))
    result={'dataset_id':dataset,'sample_id':sample,'spot_ids':spots,'spot_index':pd.Index(spots),'coords':xy,'adjacency':graph}
    if {'array_col','array_row'}<=set(coords):result['truth_coords']=coords[['array_col','array_row']].to_numpy(float)
    return result

def preprocess(sample,project):
    f=functions();coords=pd.read_csv(project/'data/coordinates'/f'{sample.dataset}.csv',dtype={'spot_id':str})
    return preprocess_coordinates(sample,project,coords)

def preprocess_coordinates(sample,project,coords):
    f=functions()
    spots=coords.spot_id.tolist();raw=project/sample.cfar_expression
    if sample.source=='internal':x,genes,aligned=f.expr.read_expression(raw.parent,spots)
    else:
        x,aligned=f.ext.read_expression(raw.parent,spots)
        # Same selected-gene rule, needed to expose truthful PCA attribution.
        data=read_expression(raw,'10x_h5')[spots];matrix=sparse.csr_matrix(data.X,dtype=np.float32)
        totals=np.asarray(matrix.sum(1)).ravel();totals[totals<=0]=1
        matrix=matrix.multiply((1e4/totals)[:,None]).tocsr();matrix.data=np.log1p(matrix.data)
        mean=np.asarray(matrix.mean(0)).ravel();var=np.maximum(np.asarray(matrix.power(2).mean(0)).ravel()-mean**2,0)
        valid=np.where(mean>.01)[0];valid=valid if len(valid) else np.arange(matrix.shape[1])
        top=valid[np.argsort(var[valid])[-min(1000,len(valid)):]];genes=data.var_names[top].tolist()
    if list(aligned)!=spots:raise ValueError('CFAR barcode order differs')
    fn=f.internal if sample.source=='internal' else f.ext
    pcs=fn.pca_scores(x);score=fn.local_scores(pcs,coords[['x','y']].to_numpy(float))
    fp=feature_pcs(project/sample.feature_expression,sample.feature_expression_kind,spots)
    # Loadings recovered from the same standardized input and PC scores, not an extra PCA fit.
    standardized=(x-np.nanmean(x,axis=0,keepdims=True))/np.maximum(np.nanstd(x,axis=0,keepdims=True),1e-3)
    loadings=(standardized.T@pcs)/np.maximum(np.sum(pcs**2,axis=0),1e-12)
    return dict(pcs=pcs,cfar=score,feature_pcs=fp,genes=np.asarray(genes,dtype=str),loadings=loadings.astype(np.float32),normalized_hvg=x,spots=np.asarray(spots,dtype=str))

def generate(sample,coords,state):
    f=functions();spots=coords.spot_id.tolist();xy=coords[['x','y']].to_numpy(float);score=state['cfar'];rows=[];seen=set()
    internal=sample.source=='internal';fn=f.internal if internal else f.ext;adj=fn.adjacency(xy) if internal else fn.radius_adjacency(xy)
    def add(indices,typ,amount,params):
        members=tuple(sorted(spots[i] for i in indices))
        if members in seen:return
        seen.add(members);cid=candidate_identity(sample.dataset,typ,members)
        rows.append({'dataset_id':sample.dataset,'sample_id':sample.sample,'candidate_id':cid,'candidate_row_key':cid,'candidate_key':sample.dataset+'||'+typ+'||'+';'.join(members),'proposal_type':typ,'member_spots':';'.join(members),'canonical_member_spots':';'.join(members),'n_spots':len(members),'proposal_score':float(amount),'proposal_params':params})
    for seed in range(len(spots)):
        add([seed],'singleton',score[seed],f'seed={seed}')
        for j in (adj[seed] if internal else sorted(adj[seed])):
            if seed<j:add([seed,j],'edge_pair',(score[seed]+score[j])/2,f'edge={seed}-{j}')
        for size in [3,4,5,8,10]:
            indices=fn.grow(seed,size,score,adj)
            if len(indices)==size:add(indices,f'core_{size}',np.mean(score[indices])+score[seed],f'seed={seed};target={size}')
    frame=pd.DataFrame(rows);assert frame.candidate_id.is_unique and frame.candidate_key.is_unique
    frame['score_per_spot']=frame.proposal_score/np.sqrt(frame.n_spots.clip(lower=1))
    grouped=frame.groupby(['dataset_id','proposal_type'],sort=False)
    frame['proposal_percentile']=grouped.proposal_score.transform(f.i.percentile_rank)
    frame['score_per_spot_percentile']=grouped.score_per_spot.transform(f.i.percentile_rank)
    frame['compactness_raw']=1/np.log1p(1+frame.n_spots.clip(lower=1));frame['compactness_percentile']=grouped.compactness_raw.transform(f.i.percentile_rank)
    return f.i.add_multiscale_persistence(frame)

def expression_features(frame,ctx,state):
    f=functions();index={s:i for i,s in enumerate(ctx['spot_ids'])}
    members=[{index[s] for s in v.split(';')} for v in frame.member_spots]
    pcs=state['feature_pcs']
    edges,anomaly=f.c.edge_distances(pcs,ctx['adjacency'])
    expression=pd.DataFrame([f.c.candidate_features(ix,pcs,ctx['adjacency'],edges,anomaly) for ix in members])
    expression['candidate_id']=frame.candidate_id.to_numpy()
    return expression

def features(frame,base,ctx,state,method,expression=None):
    f=functions();frame=frame.copy();index={s:i for i,s in enumerate(ctx['spot_ids'])}
    members=[{index[s] for s in v.split(';')} for v in frame.member_spots]
    codes=f.i.prediction_codes(base,ctx['spot_ids']);boundary,rare,_,_=f.helper.raw_sensor_evidence(frame,ctx,codes)
    frame['single_boundary_mean']=boundary[:,0];frame['single_rare_cluster_support']=rare[:,0];frame['single_island_support']=(boundary[:,0]>=.75).astype(int)
    frame['single_boundary_percentile']=frame.groupby(['dataset_id','proposal_type'],sort=False).single_boundary_mean.transform(f.i.percentile_rank)
    values,counts=np.unique(codes[:,0],return_counts=True)
    label=pd.DataFrame([f.i.label_features(ix,codes[:,0],dict(zip(map(int,values),map(int,counts)))) for ix in members])
    label['candidate_id']=frame.candidate_id.to_numpy();frame=strict_join(frame,label,'candidate_id')
    safety=f.helper.topology_safety(set(frame.candidate_row_key),frame,members,base,ctx)
    frame['topology_valid']=[key in safety for key in frame.candidate_row_key]
    for col in ['fragmentation_risk','cross_component_risk','boundary_leakage_risk','erosion_risk','safety_score','touched_upstream_components']:
        frame[col]=[safety.get(key,{}).get(col,1.0) for key in frame.candidate_row_key]
    if expression is None:expression=expression_features(frame,ctx,state)
    frame=strict_join(frame,expression,'candidate_id')
    labels=base.to_numpy(str);sizes=Counter(labels.tolist());percent=pd.Series(list(sizes.values()),index=list(sizes)).rank(method='average',pct=True).to_dict()
    summary=f.m.map_summary(labels,ctx['adjacency'])
    context=pd.DataFrame([f.m.candidate_context(ix,labels,ctx['adjacency'],sizes,percent,summary) for ix in members]);context['candidate_id']=frame.candidate_id.to_numpy()
    frame=strict_join(frame,context,'candidate_id');frame['method']=method
    X=f.g.feature_frame(frame,sorted(frame.proposal_type.unique()))
    expected=104+len(frame.proposal_type.unique())-7
    if X.shape[1]!=expected or not np.isfinite(X.to_numpy()).all():raise ValueError(f'Invalid feature schema: expected {expected}, observed {X.shape}')
    return frame

def teacher_targets(frame,ctx,bases,protocol):
    f=functions();names=sorted(bases);sensor=np.column_stack([pd.factorize(bases[m],sort=False)[0] for m in names])
    boundary,rare,_,_=f.helper.raw_sensor_evidence(frame,ctx,sensor);work=frame.copy()
    work['sensor_boundary_mean']=boundary.mean(1);work['sensor_island_support_frac']=(boundary>=.75).mean(1);work['sensor_rare_support_frac']=rare.mean(1)
    group=work.groupby(['dataset_id','proposal_type'],sort=False)
    work['boundary_percentile']=group.sensor_boundary_mean.transform(f.i.percentile_rank);work['is_singleton']=work.n_spots.eq(1)
    work['island_support_methods']=np.rint(work.sensor_island_support_frac*len(names)).astype(int);work['rare_support_methods']=np.rint(work.sensor_rare_support_frac*len(names)).astype(int)
    work['evidence_score']=.40*work.boundary_percentile+.25*work.proposal_percentile+.20*work.sensor_island_support_frac+.10*work.sensor_rare_support_frac+.05*work.compactness_percentile
    work=f.b71.apply_relaxed_evidence_gate(work,protocol);prepared,_=f.b71.prepare_dataset_candidates(work,ctx);keys={};audits=[]
    for method,base in bases.items():
        scored,_=f.b71.score_method_candidates(prepared,base,ctx,protocol)
        accepted,_,audit=f.b71.accept_scored_candidates(scored,base,ctx,method,float(protocol['budget']['primary_maximum_unique_modified_fraction']))
        if len(accepted):
            nonfrag=accepted.fragmentation_risk.le(1e-12)&accepted.erosion_risk.le(1/3+1e-12)
            frag=accepted.fragmentation_risk.gt(1e-12)&accepted.erosion_risk.le(.05+1e-12)&accepted.boundary_leakage_risk.le(1/3+1e-12)&accepted.evidence_score.ge(.65-1e-12)
            keys[method]=set(accepted.loc[nonfrag|frag,'candidate_id'])
        else:keys[method]=set()
        audits.append(dict(dataset=ctx['dataset_id'],method=method,teacher_positive=len(keys[method]),**audit))
    return keys,pd.DataFrame(audits)

def components(truth,ctx,source):
    f=functions();graph=ctx['adjacency']
    # HER2ST 21B constructs annotated truth objects with 1.25-NN radius,
    # while 21F uses a 1.55-NN graph for inference/predicted components.
    if source=='external_second':
        if 'truth_coords' not in ctx:raise RuntimeError('HER2ST array truth coordinates are required for component evaluation')
        graph=f.b2.adjacency_at_radius(ctx['truth_coords'],f.b2.infer_radius(ctx['truth_coords']))
    regions,_,_=f.b67.label_connected_components(truth,ctx['spot_ids'],graph);rows=[]
    bad={'','nan','na','none','null','unknown','<na>','unassigned','uncertain','uncertain_overlap','undetermined'}
    for cid,members in regions.items():
        label=str(truth.loc[next(iter(members))])
        if label.lower() in bad:continue
        if source=='external_first' and label not in {'In_situ_carcinoma','Tumor_cells','Tumor_cells_spindle'}:continue
        if source=='external_second' and label not in {'invasive cancer','cancer in situ'}:continue
        rows.append({'dataset_id':ctx['dataset_id'],'sample_id':ctx['sample_id'],'truth_component_id':ctx['dataset_id']+'__'+cid,'truth_label':label,'component_size_spots':len(members),'truth_spots':sorted(members)})
    return pd.DataFrame(rows)

def utility_targets(frame,base,ctx,components):
    f=functions();sets,sizes,spot_to=f.targets.component_maps(components)
    overlap,cids=f.targets.candidate_truth_overlap(frame,sets,spot_to)
    regions,_,_=f.b67.label_connected_components(base,ctx['spot_ids'],ctx['adjacency'])
    baseline={cid:f.b67.true_best_match_from_regions(regions,ss)['iou'] for cid,ss in sets.items()}
    frame=frame.copy();frame['candidate_truth_best_IoU']=overlap;frame['candidate_truth_component_id']=cids
    frame['base_best_IoU_for_candidate_target']=[baseline.get(str(cid),0.0) for cid in cids]
    frame['utility_target']=np.maximum(0,overlap-frame.base_best_IoU_for_candidate_target.to_numpy())
    frame['signed_utility_target']=overlap-frame.base_best_IoU_for_candidate_target.to_numpy()
    return frame

def infer(frame,base,ctx,bundle,ranker,config,method,expression_only=False):
    f=functions();types=bundle['proposal_types'];teacher=bundle['teacher'].predict_proba(f.t.blind_features(frame,types))[:,1]
    utility=f.signed.signed_score(frame,bundle['signed_utility_classifier'],bundle['signed_utility_regressor'],types,bundle['harm_weight'])
    slots,_=f.e.add_utility_tail(frame,teacher,utility,ctx['spot_ids'],method,bundle['extra_utility_budget'])
    if slots.empty:slots=pd.DataFrame(columns=['dataset_id','stage','proposal_type','n_spots'])
    ranking=f.k.predict(frame,ranker,types)
    ranking.insert(0,'candidate_key',frame.candidate_key.to_numpy())
    ranking.insert(0,'candidate_id',frame.candidate_id.to_numpy())
    score=frame.score_per_spot_percentile.to_numpy() if expression_only else config['expression_weight']*frame.score_per_spot_percentile.to_numpy()+config['expected_gain_weight']*ranking.score_expected_gain_rank.to_numpy()-config['cross_component_penalty']*frame.cross_component_risk.to_numpy()
    cohort='external' if bundle['held_out']=='FULL' else 'development'
    _,selected,search=f.sel.exact_token_assignment(frame,score,score,slots,ctx['spot_ids'],method,config['selector_variants'][cohort])
    budget=math.floor(config['retention']*selected.n_spots.sum()+1e-12);chosen=[];used=0
    for row in selected.sort_values(['selection_score','candidate_key'],ascending=[False,True]).itertuples():
        if used+row.n_spots<=budget:chosen.append(row.Index);used+=row.n_spots
    retained=selected.loc[chosen].copy();after=base.copy()
    for n,row in enumerate(retained.itertuples(),1):after.loc[row.member_spots.split(';')]=f'RadarST_{method}_{n}'
    return slots,retained,after,ranking,search

def evaluate(base,after,truth,ctx,component,method,cohort):
    f=functions();before_regions,_,_=f.b67.label_connected_components(base,ctx['spot_ids'],ctx['adjacency']);after_regions,_,_=f.b67.label_connected_components(after,ctx['spot_ids'],ctx['adjacency']);events=[]
    for row in component.itertuples():
        ts=set(row.truth_spots);b=f.b67.true_best_match_from_regions(before_regions,ts);a=f.b67.true_best_match_from_regions(after_regions,ts)
        events.append({'dataset':ctx['dataset_id'],'sample':ctx['sample_id'],'cohort':cohort,'method':method,'truth_component_id':row.truth_component_id,'truth_label':row.truth_label,'truth_size':len(ts),'before':b['iou'],'after':a['iou'],'gain':a['iou']-b['iou'],'coverage_before':b['overlap']/len(ts),'coverage_after':a['overlap']/len(ts),'purity_before':b['overlap']/b['region_size'] if b['region_size'] else 0.,'purity_after':a['overlap']/a['region_size'] if a['region_size'] else 0.,'dilution_before':b['region_size']/len(ts),'dilution_after':a['region_size']/len(ts)})
    bm=f.b67.strict_metrics(truth,base);am=f.b67.strict_metrics(truth,after)
    macro={'dataset':ctx['dataset_id'],'sample':ctx['sample_id'],'cohort':cohort,'method':method,'modified_spots':int((base.to_numpy()!=after.to_numpy()).sum()),'n_spots':len(base)}
    for k in ('ARI','NMI','AMI'):macro[k+'_before']=bm[k];macro[k+'_after']=am[k];macro[k+'_delta']=am[k]-bm[k]
    return pd.DataFrame(events),macro
