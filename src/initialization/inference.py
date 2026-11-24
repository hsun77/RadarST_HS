"""Standalone inference from raw expression, coordinates, one upstream map and new models."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,time
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from . import core


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for b in iter(lambda:handle.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def execute(expression,coordinates,upstream,model_dir,output,dataset,sample,source,method,barcode_col,label_col,config,feature_expression=None,feature_kind='10x_h5'):
    started=time.time();output=Path(output)
    if output.exists() and any(output.iterdir()):raise RuntimeError('Independent inference requires an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    expression=Path(expression).resolve();coordinates=Path(coordinates).resolve();upstream=Path(upstream).resolve();model_dir=Path(model_dir).resolve()
    if expression.name!='filtered_feature_bc_matrix.h5':raise ValueError('Formal CFAR loader requires filtered_feature_bc_matrix.h5')
    coords=pd.read_csv(coordinates,dtype={'spot_id':str})
    if not {'spot_id','x','y'}<=set(coords):raise ValueError('Coordinates require spot_id,x,y')
    if coords.spot_id.duplicated().any() or not np.isfinite(coords[['x','y']].to_numpy(float)).all():raise ValueError('Invalid coordinates')
    ids=set(core.read_expression(expression,'10x_h5').obs_names)
    if set(coords.spot_id)-ids:raise ValueError('Unaligned expression/coordinates')
    feature=Path(feature_expression).resolve() if feature_expression else expression
    spec=SimpleNamespace(dataset=dataset,sample=sample,source=source,cfar_expression=expression,feature_expression=feature,feature_expression_kind=feature_kind)
    bundle=joblib.load(model_dir/'selector_heads.joblib');ranker=joblib.load(model_dir/'expected_gain.joblib')
    with threadpool_limits(limits=config['blas_threads'],user_api='blas'),threadpool_limits(limits=config['openmp_threads'],user_api='openmp'):
        state=core.preprocess_coordinates(spec,Path('.'),coords);np.savez_compressed(output/'preprocess.npz',**state)
    with threadpool_limits(limits=1):
        ctx=core.make_context(dataset,sample,coords,source);base=core.read_upstream(upstream,barcode_col,label_col,ctx['spot_ids'])
        pool=core.generate(spec,coords,state);pool.to_parquet(output/'candidates.parquet',index=False)
        frame=core.features(pool,base,ctx,state,method);frame.to_parquet(output/'features.parquet',index=False)
        slots,regions,labels,ranking,search=core.infer(frame,base,ctx,bundle,ranker,config,method)
        slots.to_csv(output/'selector_slots.csv',index=False);regions.to_csv(output/'selected_regions.csv',index=False);ranking.to_parquet(output/'ranking.parquet',index=False)
        pd.DataFrame({'spot_id':ctx['spot_ids'],'pred_label':labels.to_numpy()}).to_csv(output/'final_labels.csv',index=False)
    manifest={'inputs':{str(p):digest(p) for p in [expression,feature,coordinates,upstream,model_dir/'selector_heads.joblib',model_dir/'expected_gain.joblib']},'dataset':dataset,'method':method,'configuration':config,'seconds':time.time()-started,'truth_annotation_read':False,'historical_candidates_read':False,'feature_cache_read':False,'accepted_records_read':False,'hashseed':os.environ.get('PYTHONHASHSEED','UNRECORDED'),'search':search}
    (output/'EXECUTION.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8');return manifest


def main():
    p=argparse.ArgumentParser()
    for key in ['expression','coordinates','upstream','model-dir','output','dataset','method','source','barcode-col','label-col','config']:p.add_argument('--'+key,required=True)
    p.add_argument('--sample');p.add_argument('--feature-expression');p.add_argument('--feature-kind',default='10x_h5');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());result=execute(args.expression,args.coordinates,args.upstream,args.model_dir,args.output,args.dataset,args.sample or args.dataset,args.source,args.method,args.barcode_col,args.label_col,config,args.feature_expression,args.feature_kind)
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__':main()
