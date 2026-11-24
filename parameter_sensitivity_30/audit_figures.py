"""Audit final revised PDFs; retain the manually authored plot layout."""
import os,sys,json,subprocess,argparse
from pathlib import Path
W=Path('H:/RadarST_FigureRevision2_20261111');OLD=Path('H:/RadarST_FigureCorrection_20261111');SK=Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
env=os.environ.copy();env['PYTHONPATH']=str(OLD/'deps')+os.pathsep+env.get('PYTHONPATH','');env['PYTHONIOENCODING']='utf-8'
ap=argparse.ArgumentParser();ap.add_argument('figure');a=ap.parse_args();name=a.figure
sys.path.insert(0,str(OLD/'source_code'))
import correct_figures as geometry
import numpy as np
from lxml import etree as E
tree=E.parse(str(W/'figures'/(name+'.svg')))
canvas=list(map(float,tree.getroot().get('viewBox').split()))
def plot_rectangle(panel_id,clip_id):
 clip=geometry.ident(tree,clip_id)[0]
 if clip.tag==geometry.S+'rect':
  x0,y0=float(clip.get('x')),float(clip.get('y'));x1=x0+float(clip.get('width'));y1=y0+float(clip.get('height'))
 else:x0,x1,y0,y1=geometry.bbox(clip)
 elements=tree.xpath('//*[@clip-path=$v and not(ancestor::s:defs)]',v='url(#'+clip_id+')',namespaces=geometry.NS);assert elements
 m=geometry.global_matrix(elements[0]);pts=np.asarray([m@np.array([x,y,1]) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]])
 return dict(id=panel_id,bbox_pt=[float(pts[:,0].min()),float(canvas[3]-pts[:,1].max()),float(pts[:,0].max()),float(canvas[3]-pts[:,1].min())])
if name=='Figure5':
 panels=[plot_rectangle('a-development','pe3323a8359'),plot_rectangle('a-external','p4fe44e81b1'),plot_rectangle('b-gain','pc1fa67d900'),plot_rectangle('b-minimum','p9a62b427f0')]
 groups=[['a-development','a-external'],['b-gain','b-minimum']]
else:
 panels=[]
 for k in [1,2]:
  axis=geometry.ident(tree,f'axes_{k}');target=next(p for p in geometry.paths(axis) if p.get('clip-path'));panels.append(plot_rectangle(f'upper-{k}',target.get('clip-path')[5:-1]))
 groups=[['upper-1','upper-2']]
layout=W/'qa'/(name+'_layout.json');layout.write_text(json.dumps(dict(schema_version=1,backend='Python measurement of final SVG plot rectangles',figure=dict(width_pt=canvas[2],height_pt=canvas[3]),panels=panels,row_groups=groups,exemptions=[],scope='Final comparable plot rectangles. Remaining original authored panels are checked separately by geometry and byte-preservation verification.'),indent=2),encoding='utf-8')
results=[]
for tool,args in [('audit_panel_alignment.py',[str(layout),'--json']),('audit_pdf_text.py',[str(W/'figures'/(name+'.pdf')),'--json']),('audit_figure_collisions.py',[str(W/'figures'/(name+'.pdf')),'--json'])]:
 r=subprocess.run([sys.executable,str(SK/tool),*args],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env);out=W/'qa'/f'{name}_{tool[:-3]}.json';out.write_text(r.stdout or json.dumps(dict(stderr=r.stderr)),encoding='utf-8')
 try:v=json.loads(r.stdout)
 except Exception:v=dict(error=r.stderr[-400:])
 results.append(dict(tool=tool,exit_code=r.returncode,summary=v.get('summary'),verdict=v.get('verdict'),auditable=v.get('auditable')))
(W/'qa'/f'{name}_AUTOMATED_AUDIT_SUMMARY.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results))
