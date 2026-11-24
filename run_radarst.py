"""RadarST inference from an annotation-free prepared candidate state."""
from pathlib import Path
import os,sys,json,hashlib,argparse
sys.dont_write_bytecode=True
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--molecular-dir',required=True);ap.add_argument('--output-root',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--workflow',required=True);a=ap.parse_args()
    pkg=Path(__file__).resolve().parent;inp=Path(a.input_dir).resolve();mol=Path(a.molecular_dir).resolve();out=Path(a.output_root).resolve()
    assert not out.exists() and not out.is_relative_to(pkg),'Use a new output directory outside the package.'
    out.mkdir(parents=True);(out/'tmp').mkdir()
    for k in ['TMP','TEMP','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(out/'tmp')
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
    def digest(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(8388608),b''):h.update(b)
        return h.hexdigest()
    def put(p,v):
        p=Path(p);assert p.resolve().is_relative_to(out) and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=lambda x:x.item(),allow_nan=False)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            if os.fsdecode(args[0]).lower().rstrip('\\/').split('\\')[-1]=='nul':return
            p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3];writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not p.is_relative_to(out):raise RuntimeError('WRITE_OUTSIDE_OUTPUT '+str(p))
            if not writing and (any(s.lower() in ['targets','onpolicy_targets','evaluation','external_evaluation'] for s in p.parts) or any(s in p.name.lower() for s in ['reference','truth','oracle','counterfactual'])):raise RuntimeError('SUPERVISION_IN_INFERENCE '+str(p))
    sys.addaudithook(guard);sys.path.insert(0,str(pkg/'src/numerical_core'))
    import common
    common.put=put
    for k in ['TMP','TEMP','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(out/'tmp')
    import event_round2
    event_round2.R2=pkg/'configs'
    import numpy as np,pandas as pd,joblib
    from object_evidence import assemble
    from infer_fixed_external_event import infer
    with np.load(inp/'MAP.npz',allow_pickle=False) as z:arr={k:z[k] for k in z.files}
    assert not any(s in k.lower() for k in arr for s in ['truth','reference','iou','oracle'])
    frame=pd.read_parquet(inp/'CANDIDATES.parquet');x=pd.read_parquet(inp/'BASE_FEATURES.parquet');assert x.shape[1]==147
    c=assemble(arr,frame,x);genes=pd.read_parquet(mol/'GENE_OBJECT_EVIDENCE.parquet');assert len(genes)==len(frame)
    initial=pd.read_parquet(inp/'INITIAL_SELECTED.parquet');S={c['lookup'][cid] for cid in initial.candidate_id};bundles=[]
    for n in ['utility','events']:
        p=pkg/'models'/(n+'.joblib');g=json.loads((pkg/'models'/(n+'_provenance.json')).read_text(encoding='utf-8'));assert digest(p)==g['model_sha256']
        b=joblib.load(p);b['sha256']=g['model_sha256'];assert len(b['columns'])==201 and len(b['training_datasets'])==12 and a.dataset not in b['training_datasets'];bundles.append(b)
    infer(a.dataset,a.workflow,*bundles,None,outroot=out/'predictions',cold=(c,genes,S))
    put(out/'RUN_MANIFEST.json',dict(method='RadarST',policy='E2_EVENT_CREDIT',dataset=a.dataset,workflow=a.workflow,reference_used=False,files={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and not p.is_relative_to(out/'tmp')}))
    print(out/'predictions'/a.dataset/a.workflow/'E2_EVENT_CREDIT/FINAL_LABELS.csv')
if __name__=='__main__':main()
