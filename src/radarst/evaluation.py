"""Independent connected-component evaluation using original metric functions."""
import json
from pathlib import Path


def evaluate(args):
    import numpy as np
    import pandas as pd
    from .config import new_output
    from .engine import _load
    out=new_output(args.output)
    f=_load("metrics")
    coords=pd.read_csv(args.coordinates,dtype={"spot_id":str})
    ref=pd.read_csv(args.reference,dtype=str,keep_default_na=False)
    pred=pd.read_csv(args.predictions,dtype=str,keep_default_na=False)
    for frame in [coords,ref,pred]:
        if "spot_id" not in frame or frame.spot_id.duplicated().any():
            raise ValueError("Evaluation inputs require unique spot_id")
    if "truth_label" not in ref or "pred_label" not in pred:
        raise ValueError("Reference/prediction columns must be truth_label/pred_label")
    ids=coords.spot_id.tolist()
    if not set(ids).issubset(ref.spot_id) or not set(ids).issubset(pred.spot_id):
        raise ValueError("Evaluation annotation IDs do not cover input coordinates")
    truth=ref.set_index("spot_id").truth_label.reindex(ids)
    labels=pred.set_index("spot_id").pred_label.reindex(ids)
    adj,_=f.infer_adjacency(coords[["x","y"]].to_numpy(float),args.graph_scale)
    tc=coords
    if args.reference_coordinates:
        tc=pd.read_csv(args.reference_coordinates,dtype={"spot_id":str})
        if tc.spot_id.duplicated().any() or not set(ids).issubset(tc.spot_id):
            raise ValueError("Invalid reference-coordinate IDs")
        tc=tc.set_index("spot_id").reindex(ids)
    tadj,_=f.infer_adjacency(tc[["x","y"]].to_numpy(float),args.reference_graph_scale)
    truths,_,_=f.label_connected_components(truth,ids,tadj)
    predictions,_,_=f.label_connected_components(labels,ids,adj)
    rows=[]
    for cid,members in truths.items():
        name=cid.split("::cc")[0]
        if name.lower() in f.BAD_TRUTH or args.positive_label and name not in args.positive_label:
            continue
        match=f.true_best_match_from_regions(predictions,members)
        rows.append(dict(truth_component_id=cid,truth_label=name,n_spots=len(members),best_IoU=match["iou"],coverage=match["overlap"]/len(members),matched_region_purity=match["overlap"]/max(1,match["region_size"]),matched_region_size=match["region_size"]))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"component_metrics.csv",index=False)
    summary=f.strict_metrics(truth,labels)
    summary={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in summary.items()}
    summary.update(reference_components=len(rows),compact_components=sum(1<=r["n_spots"]<=10 for r in rows),graph_scale=args.graph_scale,reference_graph_scale=args.reference_graph_scale)
    summary["mean_compact_best_IoU"]=float(frame.loc[frame.n_spots.between(1,10),"best_IoU"].mean()) if rows else None
    (out/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(status="completed",output=str(out))))
