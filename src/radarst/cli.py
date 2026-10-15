"""Public command-line entrypoints; all inputs are explicit user paths."""
import argparse
import json
import os
from pathlib import Path
import re
import sys


def parser():
    p=argparse.ArgumentParser(prog="radarst",description="Post-clustering refinement for spatial transcriptomics")
    p.add_argument("--version",action="version",version="radarst 0.1.0")
    sub=p.add_subparsers(dest="command",required=True)
    for command in ["infer","prepare-features"]:
        s=sub.add_parser(command,help="Run model-based refinement" if command=="infer" else "Generate fresh proposals and features without a model")
        for flag in ["expression","coordinates","upstream-labels","dataset","workflow","output"]:
            s.add_argument("--"+flag,required=True)
        s.add_argument("--config")
        s.add_argument("--feature-expression")
        s.add_argument("--feature-kind",choices=["h5ad","10x_h5"],default="h5ad")
        s.add_argument("--barcode-column",default="spot_id")
        s.add_argument("--label-column",default="pred_label")
        s.add_argument("--allow-positional-upstream",action="store_true")
        s.add_argument("--audit-access",action="store_true")
        s.add_argument("--ranker-model")
        s.add_argument("--selector-model")
    train=sub.add_parser("train",help="Fit the original models from explicitly prepared development supervision")
    train.add_argument("--config",required=True)
    prep=sub.add_parser("prepare-training",help="Build new development features and original-form utility targets")
    prep.add_argument("--manifest",required=True)
    prep.add_argument("--output",required=True)
    prep.add_argument('--teacher-source',choices=['provided','generate'],default='provided')
    prep.add_argument('--held-out',help='Exclude this complete sample before any teacher/target input is opened')
    prep.add_argument('--audit-access',action='store_true')
    teacher=sub.add_parser('generate-teacher',help='Generate the original multi-workflow teacher without reading reference annotations')
    teacher.add_argument('--manifest',required=True)
    teacher.add_argument('--output',required=True)
    teacher.add_argument('--held-out')
    teacher.add_argument('--audit-access',action='store_true')
    raw_train=sub.add_parser('train-inputs',help='Fold-isolated raw-input teacher, target construction and original model fitting')
    raw_train.add_argument('--manifest',required=True)
    raw_train.add_argument('--output',required=True)
    raw_train.add_argument('--mode',choices=['loso','full','loso_and_full'],default='loso_and_full')
    raw_train.add_argument('--held-out',help='Run one LOSO fold only; requires --mode loso')
    raw_train.add_argument('--audit-access',action='store_true')
    ev=sub.add_parser("evaluate",help="Evaluate predictions independently of inference")
    for flag in ["coordinates","reference","predictions","output"]: ev.add_argument("--"+flag,required=True)
    ev.add_argument("--graph-scale",type=float,default=1.25)
    ev.add_argument("--reference-coordinates",help="Optional spot_id,x,y CSV for a separate reference graph")
    ev.add_argument("--reference-graph-scale",type=float,default=1.25)
    ev.add_argument("--positive-label",action="append")
    syn=sub.add_parser("synthetic",help="Generate a small artificial software-test input")
    syn.add_argument("--output",required=True)
    return p


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try:
        if args.command in {"infer","prepare-features"}:
            if args.command=="infer":
                for attr in ["ranker_model","selector_model"]:
                    value=getattr(args,attr)
                    if not value or not Path(value).is_file():
                        raise ValueError("Explicit compatible --ranker-model and --selector-model files are required. No weights are bundled or auto-discovered. Use radarst train with documented development supervision; see docs/training.md.")
            if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",s) or ".." in s for s in [args.dataset,args.workflow]):
                raise ValueError("Dataset/workflow identifiers must be safe single path components")
            from .config import read_config,new_output
            config=read_config(args.config)
            if os.environ.get("PYTHONHASHSEED") != config["python_hash_seed"]:
                raise ValueError("Set PYTHONHASHSEED=0 before starting Python; changing it after startup cannot set the process hash state")
            output=new_output(args.output)
            scratch=output/"runtime_tmp"
            scratch.mkdir()
            os.environ.update(TEMP=str(scratch),TMP=str(scratch),NUMBA_CACHE_DIR=str(output/"numba_cache"),MPLCONFIGDIR=str(output/"mplconfig"))
            audit=None
            if args.audit_access:
                from .audit import install
                audit=install(output,[getattr(args,k,None) for k in ["expression","coordinates","upstream_labels","feature_expression","ranker_model","selector_model","config"]])
            from .inference import run
            run(args,output,config,args.command=="prepare-features")
            if audit: audit.flush()
            print(json.dumps(dict(status="completed",command=args.command,output=str(output))))
        elif args.command=="synthetic":
            from .synthetic import generate
            generate(args.output)
        elif args.command=="evaluate":
            from .evaluation import evaluate
            evaluate(args)
        elif args.command=="prepare-training":
            from .training import prepare
            prepare(args.manifest,args.output,args.teacher_source,args.held_out,args.audit_access)
        elif args.command=='generate-teacher':
            from .teacher import generate
            generate(args.manifest,args.output,args.held_out,args.audit_access)
        elif args.command=='train-inputs':
            from .training import train_inputs
            train_inputs(args.manifest,args.output,args.mode,args.held_out,args.audit_access)
        elif args.command=="train":
            from .training import train
            train(args.config)
    except (ValueError,FileNotFoundError,KeyError,RuntimeError) as error:
        p.exit(2,f"radarst: error: {error}\n")
