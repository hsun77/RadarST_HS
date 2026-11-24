from pathlib import Path
import os,sys,json,hashlib,csv,time,datetime
ROOT=Path('H:/RadarST_2_ObjectValue_20261104')
AF=Path('H:/RadarST_Next_AccuracyFirst_20261023')
EX=Path('H:/RadarST_2_ExactMask_20261028')
OLD=Path('H:/RadarST_I3_REBUILD_20261021')
AUD=Path('H:/RadarST_2_StageLossAudit_20261104')
REG=Path('H:/RadarST_CleanV1_BANKSY_FULL_20261021/audits/REGISTERED_322_PAIRS.csv')
sys.dont_write_bytecode=True
for k in ['TMP','TEMP','TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER']:os.environ[k]=str(ROOT/'tmp')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
 p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve()) and not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True)
 text=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item())
 q=p.with_name(p.name+'.partial-'+str(os.getpid()))
 with q.open('x',encoding='utf8') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.rename(q,p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pairs():
 with REG.open(encoding='utf-8-sig',newline='') as f:return sorted([r for r in csv.DictReader(f) if r['cohort']=='development'],key=lambda r:(r['dataset'],r['workflow']))
def guard(role='observable',heldout=None):
 regs=pairs();refs={str(Path(r[k]).resolve()).lower():r['dataset'] for r in regs for k in ['reference','metadata'] if r.get(k)}
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   raw=os.fsdecode(args[0])
   if raw.lower()=='nul':return
   p=Path(raw).resolve();mode,flags=args[1:3];write=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
   if write:
    if not p.is_relative_to(ROOT.resolve()):raise RuntimeError('WRITE_OUTSIDE_NEW_RESEARCH '+str(p))
    return
   key=str(p).lower();rel=p.as_posix().lower()
   if role in ['inference','observable'] and key in refs:raise RuntimeError('ANNOTATION_IN_DEPLOYABLE_STAGE '+str(p))
   if role=='training' and key in refs and refs[key]==heldout:raise RuntimeError('OUTER_LABEL_READ '+str(p))
   if role in ['training','inference','observable']:
    if p.is_relative_to(ROOT/'evaluation') or p.is_relative_to(AUD) or any(x in rel for x in ['oracle','counterfactual','component_transitions']):raise RuntimeError('OFFLINE_RESULT_IN_LEARNING '+str(p))
   if role in ['inference','observable'] and (p.is_relative_to(ROOT/'targets') or p.is_relative_to(AF/'targets') or p.is_relative_to(AF/'onpolicy_targets') or p.is_relative_to(EX/'targets')):raise RuntimeError('TARGET_READ_IN_DEPLOYMENT '+str(p))
   if role=='training' and heldout:
    if p.is_relative_to(AF/'targets'/heldout):raise RuntimeError('OUTER_TARGET_IN_TRAINING '+str(p))
    if p.is_relative_to(ROOT/'targets'):
     pp=p.relative_to(ROOT/'targets').parts
     if len(pp)>=2 and pp[1]=='nested':
      if pp[0]!=heldout or (len(pp)>=3 and pp[2]==heldout):raise RuntimeError('WRONG_NEW_NESTED_TARGET_ROUTE '+str(p))
     elif pp and pp[0]==heldout:raise RuntimeError('OUTER_TARGET_IN_TRAINING '+str(p))
    if p.is_relative_to(AF/'onpolicy_targets'):
     parts=p.relative_to(AF/'onpolicy_targets').parts
     if len(parts)>=2 and (parts[0]!=heldout or parts[1]==heldout):raise RuntimeError('WRONG_NESTED_TARGET_ROUTE '+str(p))
    if p.is_relative_to(AF/'onpolicy_states'):
     parts=p.relative_to(AF/'onpolicy_states').parts
     if parts and heldout not in parts[0].split('__'):raise RuntimeError('WRONG_NESTED_STATE_ROUTE '+str(p))
  if event in ['os.mkdir','os.remove','os.rmdir','os.rename']:
   for raw in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(raw,(str,bytes)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(ROOT.resolve()):raise RuntimeError('OUTSIDE_MUTATION '+str(raw))
 sys.addaudithook(hook)
class Verify:
 def __init__(self):self.man={r:js(r/'SHA256_MANIFEST.json')['files'] for r in [AF,EX]};self.checked={}
 def check(self,p,expected=None):
  p=Path(p);key=str(p)
  if key in self.checked:
   if expected is not None:assert self.checked[key]==expected
   return self.checked[key]
  h=sha(p)
  if expected is None:
   r=next((r for r in self.man if p.is_relative_to(r)),None);assert r,p
   e=self.man[r][p.relative_to(r).as_posix()];expected=e['sha256'] if isinstance(e,dict) else e
  assert h==expected,('SOURCE_HASH_MISMATCH',p)
  self.checked[key]=h;return h
