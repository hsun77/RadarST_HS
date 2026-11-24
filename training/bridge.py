"""Output-only binding to frozen numerical modules; original files stay read-only."""
import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SOURCE = Path('H:/RadarST_2_ObjectValue_20261104')
REG = Path('H:/RadarST_CleanV1_BANKSY_FULL_20261021/audits/REGISTERED_322_PAIRS.csv')
PYTHON = Path('C:/Environment_Langguage_Software/Anaconda/envs/st_benchmark/python.exe')
sys.dont_write_bytecode = True
for k in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER']:
    os.environ[k] = str(ROOT/'tmp')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()

def js(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def utc(): return datetime.now(timezone.utc).isoformat()
def put(p,v):
    p=Path(p); assert p.resolve().is_relative_to(ROOT.resolve()) and not p.exists(), p
    p.parent.mkdir(parents=True,exist_ok=True)
    text=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())
    q=p.with_name(p.name+'.partial-'+str(os.getpid()))
    with q.open('x',encoding='utf-8',newline='') as f: f.write(text);f.flush();os.fsync(f.fileno())
    q.rename(p)

def pairs(cohort):
    with REG.open(encoding='utf-8-sig',newline='') as f:
        return sorted([r for r in csv.DictReader(f) if r['cohort']==cohort],key=lambda r:(r['dataset'],r['workflow']))

def frozen(p):
    p=Path(p)
    if not hasattr(frozen,'manifest'):
        receipt=js(SOURCE/'FINAL_SEAL_RECEIPT.json')
        assert sha(SOURCE/'FINAL_SHA256_MANIFEST.json')==receipt['manifest_sha256']
        frozen.manifest=js(SOURCE/'FINAL_SHA256_MANIFEST.json')['files']
    h=sha(p); assert h==frozen.manifest[p.relative_to(SOURCE).as_posix()]['sha256'], p
    return h

def guard(mode):
    regs=pairs('development')+pairs('external')
    references={str(Path(r[k]).resolve()).lower() for r in regs for k in ['reference','metadata'] if r.get(k)}
    extroots=[SOURCE/'external_inputs', SOURCE/'external_predictions', SOURCE/'external_evaluation',
              Path('H:/RadarST_2_ExactMask_External_FULL_20261030'),Path('H:/RadarST_Next_CET_External_20261025')]
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            raw=os.fsdecode(args[0])
            if raw.lower()=='nul':return
            p=Path(raw).resolve();m,flags=args[1:3]
            writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing:
                if not p.is_relative_to(ROOT):raise RuntimeError('WRITE_OUTSIDE_NEW_EXPERIMENT '+str(p))
                return
            name=p.as_posix().lower();parts=[s.lower() for s in p.parts]
            if mode=='training' and (str(p).lower() in references or any(p.is_relative_to(r) for r in extroots)):
                raise RuntimeError('EXTERNAL_OR_REFERENCE_READ_DURING_FULL_FIT '+str(p))
            if mode=='inference':
                if str(p).lower() in references:raise RuntimeError('REFERENCE_READ_DURING_INFERENCE '+str(p))
                if any(s in ['targets','onpolicy_targets','evaluation','external_evaluation','offline_diagnostics','offline_oracles','training'] or s.startswith('full_fit_targets') or s.startswith('full_fit_context_targets') or s.startswith('full_fit_object_targets') or s.startswith('full_fit_event_targets') for s in parts):
                    raise RuntimeError('SUPERVISION_OR_ENDPOINT_READ_DURING_INFERENCE '+str(p))
                if 'oracle' in name or 'counterfactual' in name or 'component_transitions' in name:
                    raise RuntimeError('OFFLINE_READ_DURING_INFERENCE '+str(p))
        if event in ['os.mkdir','os.remove','os.rmdir','os.rename']:
            for raw in args[:2] if event=='os.rename' else args[:1]:
                if isinstance(raw,(str,bytes)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(ROOT):
                    raise RuntimeError('MUTATION_OUTSIDE_NEW_EXPERIMENT '+str(raw))
    sys.addaudithook(hook)

def bind():
    # Only redirect write helper and environment. Numerical ROOT stays the sealed source,
    # and each inference call receives an explicit new outroot and observable inputs.
    sys.path.insert(0,str(SOURCE/'code'))
    import common
    common.put=put
    for k in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER']:
        os.environ[k]=str(ROOT/'tmp')
    return common

def check_lock():
    g=js(ROOT/'PROTOCOL_LOCK.json')
    assert g['method']=='E2_EVENT_CREDIT' and sha(REG)==g['registry_sha256']
    for name,h in g['new_code_sha256'].items():assert sha(ROOT/name)==h, name
    for name,h in g['frozen_numeric_code_sha256'].items():assert sha(SOURCE/name)==h, name
    return g
