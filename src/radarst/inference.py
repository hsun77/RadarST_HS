"""Relocatable invocation of the original detector, features and selector."""
from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import os
import shutil


def input_row(args, output, config):
    import numpy as np
    import pandas as pd
    expression = Path(args.expression).resolve()
    coords = Path(args.coordinates).resolve()
    upstream = Path(args.upstream_labels).resolve()
    for path in [expression, coords, upstream]:
        if not path.is_file():
            raise ValueError(f"Input file not found: {path}")
    c = pd.read_csv(coords, dtype={"spot_id":str})
    if not {"spot_id","x","y"}.issubset(c) or c.spot_id.isna().any() or c.spot_id.duplicated().any():
        raise ValueError("Coordinates require unique nonmissing spot_id,x,y")
    if not np.isfinite(c[["x","y"]].to_numpy(float)).all() or len(c) < 21:
        raise ValueError("Finite coordinates and at least 21 spots are required by the existing PCA protocol")
    base = pd.read_csv(upstream, dtype=str).fillna("")
    for col in [args.barcode_column, args.label_column]:
        if col not in base:
            raise ValueError(f"Missing upstream column: {col}")
    if base[args.barcode_column].duplicated().any() or base[args.label_column].eq("").any():
        raise ValueError("Duplicate upstream IDs or empty labels")
    missing = set(c.spot_id) - set(base[args.barcode_column])
    if missing and not (args.allow_positional_upstream and len(base)==len(c) and 1-len(missing)/len(c)<.5):
        raise ValueError("Upstream IDs do not align; legacy positional fallback requires explicit opt-in")
    if expression.name != "filtered_feature_bc_matrix.h5":
        folder = output / "_input"
        folder.mkdir()
        target = folder / "filtered_feature_bc_matrix.h5"
        try:
            os.link(expression,target)
        except OSError:
            shutil.copyfile(expression,target)
        expression = target
    feature = Path(args.feature_expression).resolve() if args.feature_expression else expression
    return SimpleNamespace(dataset=args.dataset, sample=args.dataset, source=config["profile"],
        method=args.workflow, coordinates=str(coords), cfar_expression=str(expression),
        expression=str(feature), expression_kind=args.feature_kind if args.feature_expression else "10x_h5",
        upstream=str(upstream), barcode_col=args.barcode_column, label_col=args.label_column,
        ranker_model=str(Path(args.ranker_model).resolve()) if getattr(args,"ranker_model",None) else "",
        selector_model=str(Path(args.selector_model).resolve()) if getattr(args,"selector_model",None) else "")


def setup(row):
    import pandas as pd
    from . import frontend as F, expression_io as E
    from .engine import functions
    if F.feature_pcs is not E.expression_only_feature_pcs:
        E.original_feature_pcs = F.feature_pcs
    E.table = pd.DataFrame([vars(row)])
    F.feature_pcs = E.expression_only_feature_pcs
    return F, functions()


def validate_models(row):
    import joblib
    rank, heads = joblib.load(row.ranker_model), joblib.load(row.selector_model)
    rk = {"positive_models","rank_models","harm_models","proposal_types","feature_columns"}
    hk = {"teacher","signed_utility_classifier","signed_utility_regressor","proposal_types","harm_weight","extra_utility_budget"}
    if not isinstance(rank,dict) or not rk.issubset(rank) or len(rank["feature_columns"]) != 104:
        raise ValueError("Incompatible expected-gain model schema; see docs/models.md")
    if not isinstance(heads,dict) or not hk.issubset(heads):
        raise ValueError("Incompatible selector-head model schema; see docs/models.md")
    if heads["proposal_types"] != rank["proposal_types"] or heads["harm_weight"] != .5 or heads["extra_utility_budget"] != .01:
        raise ValueError("Incompatible proposal types or selector protocol")
    for model in [heads[k] for k in ["teacher","signed_utility_classifier","signed_utility_regressor"]]:
        if model.n_features_in_ != 28:
            raise ValueError("Selector heads must have the implemented 28 features")
    if row.source == "internal":
        for model in [rank,heads]:
            if model.get("held_out_dataset") != row.dataset or row.dataset in model.get("training_datasets",[]):
                raise ValueError("Internal inference requires matching held-out models, not a full-fit substitute")
    return rank,heads


def run(args, output, config, features_only=False):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from .engine import _load
    row = input_row(args,output,config)
    models = None if features_only else validate_models(row)
    F,f = setup(row)
    captured = {}
    selector = f.sel.exact_token_assignment
    def capture(*a,**kw):
        result=selector(*a,**kw)
        captured["full_selected"]=result[1].copy()
        return result
    f.sel.exact_token_assignment = capture
    with threadpool_limits(limits=20,user_api="blas"), threadpool_limits(limits=1,user_api="openmp"):
        candidates,ctx,indices,scores=F.generate_sample(row,f)
        candidates.to_parquet(output/"candidates.parquet",index=False)
        pd.DataFrame({"spot_id":ctx["spot_ids"],"CFAR":scores}).to_csv(output/"CFAR_scores.csv",index=False)
        if features_only:
            frame,_=F.build_pair_features(row,candidates,ctx,indices,f,output)
            frame.to_parquet(output/"features.parquet",index=False)
            x=f.g.feature_frame(frame,["core_10","core_3","core_4","core_5","core_8","edge_pair","singleton"])
            x.to_parquet(output/"model_feature_array.parquet",index=False)
        else:
            rank,heads=models
            F.generate_pair(row,candidates,ctx,indices,f,output)
            dest=output/row.dataset/row.method
            frame=pd.read_parquet(dest/"features.parquet")
            x=f.t.blind_features(frame,heads["proposal_types"])
            teacher=heads["teacher"].predict_proba(x)[:,1]
            positive=heads["signed_utility_classifier"].predict_proba(x)[:,1]
            regression=heads["signed_utility_regressor"].predict(x)
            utility=f.signed.signed_score(frame,heads["signed_utility_classifier"],heads["signed_utility_regressor"],heads["proposal_types"],heads["harm_weight"])
            pd.DataFrame({"candidate_key":frame.candidate_key,"teacher_probability":teacher,"signed_positive_probability":positive,"signed_regression":regression,"signed_score":utility}).to_parquet(dest/"three_head_predictions.parquet",index=False)
            if row.source=="internal":
                accepted,_=f.e.add_utility_tail(frame,teacher,utility,ctx["spot_ids"],row.method,heads["extra_utility_budget"])
                if accepted.empty:
                    accepted=pd.DataFrame(columns=["dataset_id","sample_id","method","candidate_key","candidate_id","stage","proposal_type","n_spots","member_spots","selection_score"])
                accepted.to_csv(dest/"selector_slots_generated.csv",index=False)
                combined=pd.read_parquet(dest/"ranking.parquet").selection_score.to_numpy(float)
                _,selected,search=f.sel.exact_token_assignment(frame,combined,combined,accepted,ctx["spot_ids"],row.method,"anchor_cross_0.025")
                assignment,count,used,budget=_load("retention").retained_assignment(selected,config["retention"],row.dataset,row.method)
                retained=selected[selected.member_spots.map(lambda text: all(s in assignment for s in str(text).split(";") if s))].copy()
                retained.to_csv(dest/"selected_regions.csv",index=False)
                base=f.b67.load_prediction(Path(row.upstream),pd.Index(ctx["spot_ids"]),row.barcode_col,row.label_col)
                final=base.copy()
                for spot,label in assignment.items(): final.loc[spot]=label
                pd.DataFrame({"spot_id":ctx["spot_ids"],"pred_label":final.reindex(ctx["spot_ids"]).to_numpy(str)}).to_csv(dest/"final_labels.csv",index=False)
            captured["full_selected"].to_csv(dest/"full_selected_regions.csv",index=False)
            status=json.loads((dest/"status.json").read_text())
            status.update(template_status="GENERATED_FROM_EXPLICIT_USER_SUPPLIED_SELECTOR_MODEL",completed=True,selector_model_sha256=F.sha(row.selector_model))
            F.dump(dest/"status.json",status)
            shutil.copyfile(dest/"final_labels.csv",output/"refined_labels.csv")
            shutil.copyfile(dest/"selected_regions.csv",output/"selected_regions.csv")
    inputs={k:dict(path=str(Path(v).resolve()),sha256=F.sha(v)) for k,v in vars(row).items() if k in {"coordinates","cfar_expression","expression","upstream","ranker_model","selector_model"} and v}
    metadata=dict(command="prepare-features" if features_only else "infer",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=row.dataset,workflow=row.method,profile=row.source,configuration=config,inputs=inputs,python_hash_seed=os.environ.get("PYTHONHASHSEED"),supervised_training_performed=False,reference_annotations_read=False,feature_columns=list(f.g.feature_frame(frame,["core_10","core_3","core_4","core_5","core_8","edge_pair","singleton"]).columns))
    F.dump(output/"run_metadata.json",metadata)
    F.dump(output/"configuration.json",config)
    return output
