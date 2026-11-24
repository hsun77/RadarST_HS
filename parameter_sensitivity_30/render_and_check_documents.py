"""Render edited documents; reuse byte-verified prior baseline renders."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,zipfile,time
from PIL import Image,ImageDraw
W=Path('H:/RadarST_FigureRevision2_20261111');OLD=Path('H:/RadarST_FigureCorrection_20261111');R=Path('C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/documents/26.1104.22356/skills/documents/render_docx.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=os.environ.copy();env['PATH']='C:/Program Files/LibreOffice/program;C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/native/poppler/Library/bin;'+env['PATH'];env['TEMP']=env['TMP']=str(W/'tmp')
before={r['document']:r['staged_sha256'] for r in json.loads((OLD/'qa/DOCX_MEDIA_ONLY_AUDIT.json').read_text())};records=[]
for name in ['manuscript_marked','Supplementary_Information','Supplementary_Information_marked']:
 assert sha(W/'before'/(name+'.docx'))==before[name+'.docx'];out=W/'qa'/('staged_'+name)
 render_gate=out/'RENDER_GATE.json';current_sha=sha(W/'staged'/(name+'.docx'));valid=False
 if render_gate.exists() and (out/(name+'.pdf')).exists():
  g=json.loads(render_gate.read_text());valid=g['docx_sha256']==current_sha and g['pdf_sha256']==sha(out/(name+'.pdf'))
 if not valid:
  if out.exists():
   archive=out.with_name(out.name+'_previous_'+str(time.time_ns()));assert out.resolve().is_relative_to((W/'qa').resolve()) and archive.resolve().is_relative_to((W/'qa').resolve());out.rename(archive)
  r=subprocess.run([sys.executable,str(R),str(W/'staged'/(name+'.docx')),'--output_dir',str(out),'--emit_pdf'],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env);(W/'qa'/('render_'+name+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');assert r.returncode==0,(name,r.stdout,r.stderr)
  render_gate.write_text(json.dumps(dict(docx_sha256=current_sha,pdf_sha256=sha(out/(name+'.pdf'))),indent=2),encoding='utf-8')
 pages=sorted(out.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]));base=sorted((OLD/'qa'/('staged_'+name)).glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]));changed=[i+1 for i,p in enumerate(pages) if i>=len(base) or sha(p)!=sha(base[i])]
 for start in range(0,len(pages),12):
  height=350*((min(12,len(pages)-start)+3)//4);canvas=Image.new('RGB',(1000,height),'#dddddd');draw=ImageDraw.Draw(canvas)
  for j,p in enumerate(pages[start:start+12]):
   im=Image.open(p).convert('RGB');im.thumbnail((245,325));x=(j%4)*250+(250-im.width)//2;y=(j//4)*350+20;canvas.paste(im,(x,y));draw.text(((j%4)*250+5,(j//4)*350+2),f'{name} p{start+j+1}'+(' UPDATED' if start+j+1 in changed else ''),fill='black')
  canvas.save(W/'qa'/f'{name}_contact_{start+1}.png')
 with zipfile.ZipFile(W/'staged'/(name+'.docx')) as z:
  if name=='manuscript_marked':
   for k in [3,5]:assert z.read(f'word/media/image{k}.png')==(W/'figures'/f'Figure{k}.png').read_bytes()
 record=dict(document=name+'.docx',before_pages=len(base),after_pages=len(pages),changed_pages=changed,docx_sha256=sha(W/'staged'/(name+'.docx')),pdf_sha256=sha(out/(name+'.pdf')),before_render_reused_after_exact_document_hash_verification=True)
 records.append(record);print(json.dumps(record),flush=True)
(W/'qa/DOCUMENT_RENDER_COMPARISON.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
