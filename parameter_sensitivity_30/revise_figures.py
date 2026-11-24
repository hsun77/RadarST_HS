"""Targeted Figure3 axes/connection and Figure5 original-style corrections."""
import sys,copy,json,hashlib,subprocess,shutil,inspect
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'H:/RadarST_FigureCorrection_20261111/source_code')
import correct_figures as b
import numpy as np,pandas as pd
from lxml import etree as E
from PIL import Image,ImageDraw,ImageFilter
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import fitz
W=Path('H:/RadarST_FigureRevision2_20261111');D=Path('H:/RadarST_PROJECT/NC_SUBMISSION_FINAL');ORIG=Path('G:/RadarST_PROJECT/01_manuscript/current_submission/figures')
for p in ['before/figures','figures','data','qa','staged']:(W/p).mkdir(parents=True,exist_ok=True)
def backup():
 for n in [3,5]:
  for ext in ['svg','pdf','png']:
   dest=W/'before/figures'/f'Figure{n}.{ext}'
   if not dest.exists():shutil.copy2(D/'Main_Figures'/f'Figure{n}.{ext}',dest)
 for name in ['manuscript_marked.docx','manuscript_marked.pdf','Supplementary_Information.docx','Supplementary_Information_marked.docx','Supplementary_Tables.xlsx','DELIVERY_COMPLETE.json','DELIVERY_SHA256_MANIFEST.json']:
  if not (W/'before'/name).exists():shutil.copy2(D/name,W/'before'/name)
def unique_ids(g,prefix):
 lookup={x.get('id'):prefix+x.get('id') for x in g.iter() if isinstance(x.tag,str) and x.get('id')}
 for x in g.iter():
  if not isinstance(x.tag,str):continue
  if x.get('id'):x.set('id',lookup[x.get('id')])
  for k,v in list(x.attrib.items()):
   if v.startswith('#') and v[1:] in lookup:x.set(k,'#'+lookup[v[1:]])
   elif v.startswith('url(#') and v[5:-1] in lookup:x.set(k,'url(#'+lookup[v[5:-1]]+')')
def preserve_referenced_definitions(r,source):
 import re
 prior=E.parse(str(source));restored=[]
 while True:
  ids={x.get('id') for x in r.iter() if isinstance(x.tag,str) and x.get('id')};refs=[]
  for x in r.iter():
   if not isinstance(x.tag,str):continue
   for k,v in x.attrib.items():
    refs.extend(re.findall(r'url\(#([^)]+)\)',v))
    if (k.endswith('}href') or k=='href') and v.startswith('#'):refs.append(v[1:])
  missing=sorted(set(refs)-ids)
  if not missing:break
  holder=E.SubElement(r.getroot(),b.S+'defs')
  for id in missing:
   original=b.ident(prior,id);assert original.xpath('ancestor::s:defs',namespaces=b.NS),id;holder.append(copy.deepcopy(original));restored.append(id)
 return restored
def asset_preflight():
 import re,collections
 r=E.parse(str(W/'before/figures/Figure5.svg'));original=E.parse(str(ORIG/'Figure05_robustness_redesign.svg'));old=b.ident(r,'corrected_panel_b');old.getparent().replace(old,copy.deepcopy(b.ident(original,'g1201')));preserved=preserve_referenced_definitions(r,ORIG/'Figure05_robustness_redesign.svg')
 ids=collections.Counter(x.get('id') for x in r.iter() if isinstance(x.tag,str) and x.get('id'));duplicate=[k for k,v in ids.items() if v>1];refs=[]
 for x in r.iter():
  if isinstance(x.tag,str):
   for k,v in x.attrib.items():
    refs.extend(re.findall(r'url\(#([^)]+)\)',v))
    if (k.endswith('}href') or k=='href') and v.startswith('#'):refs.append(v[1:])
 missing=sorted(set(refs)-set(ids));report=dict(duplicate_ids=duplicate,missing_clip_or_style_references=missing,restored_cells_left=sum('fill:none' not in x.get('style','') for x in r.xpath('.//s:path[@clip-path="url(#pc1fa67d900)"]',namespaces=b.NS)),restored_cells_right=sum('fill:none' not in x.get('style','') for x in r.xpath('.//s:path[@clip-path="url(#p9a62b427f0)"]',namespaces=b.NS)))
 report['preserved_shared_definitions']=preserved;(W/'qa/ORIGINAL_PANEL_ASSET_PREFLIGHT.json').write_text(json.dumps(report,indent=2));assert not duplicate and not missing,report;print(json.dumps(report))
def figure3():
 r=E.parse(str(W/'before/figures/Figure3.svg'));verify=[]
 for n,cohort in [(1,'development'),(2,'external')]:
  a=b.ident(r,f'axes_{n}');axis=b.ident(r,'matplotlib.axis_'+str(2*n));data=pd.read_csv(Path('H:/RadarST_FigureCorrection_20261111/data')/f'Figure3_{cohort}_exact_size.csv').set_index('component_size_spots').reindex(range(1,11))
  line=[g for g in a if (g.get('id') or '').startswith('line2d_')];assert len(line)==2
  firstpath=b.paths(line[0])[0];clip=b.ident(r,firstpath.get('clip-path')[5:-1])[0];top=float(clip.get('y'));height=float(clip.get('height'));fy=lambda v:top+height-np.asarray(v)/.5
  # Correct formula retains the complete original plotting rectangle.
  fy=lambda v:top+height*(1-np.asarray(v)/.5)
  xall=np.array([float(u.get('x')) for u in b.uses(line[0])]);finite=np.isfinite(data.base_best_IoU.to_numpy());assert len(xall)==finite.sum()
  available=np.flatnonzero(finite);xs=np.interp(np.arange(10),available,xall)
  for g,column in zip(line,['base_best_IoU','after_best_IoU']):
   values=data[column].to_numpy();mask=np.isfinite(values);uu=b.uses(g);assert len(uu)==mask.sum()
   for u,v in zip(uu,values[mask]):b.setuse(u,y=float(fy(v)))
   # Available sizes only. In panel b, size8 connects directly to size10;
   # no estimated or invented point is created at size9.
   b.paths(g)[0].set('d',b.poly([(float(u.get('x')),float(u.get('y'))) for u in uu]))
  fill=next(g for g in a if 'FillBetween' in (g.get('id') or ''));path=fill.xpath('.//s:defs/s:path',namespaces=b.NS)[0];shift=float(b.uses(fill)[0].get('y'));valid=np.isfinite(data.base_best_IoU)&np.isfinite(data.after_best_IoU);pieces=[]
  for seq in np.split(np.arange(10),np.flatnonzero(np.diff(valid.astype(int)))+1):
   if valid.iloc[seq[0]]:pieces.append(b.poly(list(zip(xs[seq],fy(data.base_best_IoU.to_numpy()[seq])-shift))+list(zip(xs[seq[::-1]],fy(data.after_best_IoU.to_numpy()[seq[::-1]])-shift)),True))
  path.set('d',' '.join(pieces))
  ticks=[g for g in axis if (g.get('id') or '').startswith('ytick_')];template=copy.deepcopy(ticks[0]);u=b.uses(ticks[0])[0];origin=(b.global_matrix(u)@np.array([float(u.get('x')),float(u.get('y')),1]))[1]
  for g in ticks:axis.remove(g)
  for i,value in enumerate(np.linspace(0,.5,6)):
   g=copy.deepcopy(template);unique_ids(g,f'SHARED_{n}_{i}_');b.settext(b.texts(g)[0],f'{value:.1f}');b.shift_page(g,dy=float(fy(value)-origin));axis.append(g)
  assert data.after_best_IoU.max()<.5
  verify.append(dict(panel=n,ylim=[0,.5],ticks=[0,.1,.2,.3,.4,.5],data_values_unchanged=True,markers_per_series=int(finite.sum()),missing_size9_has_marker=False if n==2 else None,connect_8_to_10=n==2))
 for id in ['text_20','text_41','text_90']:b.shift_page(b.ident(r,id),dy=-5)
 restored=preserve_referenced_definitions(r,W/'before/figures/Figure3.svg');verify.append(dict(shared_marker_definitions_preserved=restored,all_SVG_references_resolve=True))
 r.write(str(W/'figures/Figure3.svg'),encoding='utf-8',xml_declaration=True);(W/'qa/FIGURE3_CHANGE_AUDIT.json').write_text(json.dumps(verify,indent=2),encoding='utf-8')
def restore_parameter_panel(r):
 summary=pd.read_csv(W/'evaluation/PARAMETER_SENSITIVITY_30.csv');assert len(summary)==30
 original=E.parse(str(ORIG/'Figure05_robustness_redesign.svg'));g=copy.deepcopy(b.ident(original,'g1201'));old=b.ident(r,'corrected_panel_b');old.getparent().replace(old,g);preserve_referenced_definitions(r,ORIG/'Figure05_robustness_redesign.svg')
 olddata=pd.read_csv(Path('G:/RadarST_PROJECT/01_manuscript/supplementary_tables/Supplementary_Table_S20_development_parameter_sensitivity.csv'))
 heat=colors.LinearSegmentedColormap.from_list('exact_original_blues',['#F1F5F8','#BFD7E8','#5B9BC8','#1E5D8E'])
 for j,(metric,oldmetric,cl,firsttext) in enumerate([('sample_equal_size_1_10_gain','sample_equal_size_1_10_gain','pc1fa67d900',57),('minimum_method_gain','minimum_method_gain','p9a62b427f0',95)]):
  table=summary.pivot(index='expression_weight',columns='retention_fraction',values=metric).reindex(index=[.90,.85,.80,.75,.70],columns=[.95,.96,.97,.98,.99,1.0]);v=table.to_numpy();assert np.isfinite(v).all();lo=float(v.min());hi=float(v.max());pad=max((hi-lo)*.06,1e-6);nlo=lo-pad;nhi=hi+pad;norm=colors.Normalize(nlo,nhi)
  cells=[p for p in g.xpath('.//*[@clip-path=$cl]',cl='url(#'+cl+')') if p.tag==b.S+'path' and 'fill:none' not in p.get('style','')];assert len(cells)==30
  cells.sort(key=lambda p:(round(b.bbox(p)[2],3),b.bbox(p)[0]))
  for i,p in enumerate(cells):
   val=v.flat[i];st=p.get('style');p.set('style',__import__('re').sub(r'fill\s*:[^;]+','fill:'+colors.to_hex(heat(norm(val))),st));text=b.texts(b.ident(g,f'text_{firsttext+i}'))[0];b.settext(text,f'{val:.3f}')
   # Original annotation text colors reflect the original color scale. Update
   # only contrast to remain legible on the corrected cell shade.
   relative=(val-lo)/max(hi-lo,1e-12);ink='white' if relative>.62 else '#1e2933';text.set('style',__import__('re').sub(r'fill\s*:[^;]+','fill:'+ink,text.get('style','')))
  oldlo=float(olddata[oldmetric].min());oldhi=float(olddata[oldmetric].max());oldpad=max((oldhi-oldlo)*.06,1e-6);oldlo-=oldpad;oldhi+=oldpad
  # plot_heatmap uses Normalize(min,max), so preserve all original colorbar
  # geometry and update only the actual numeric calibration.
  for t in b.texts(g):
   text=''.join(t.itertext()).strip()
   try:val=float(text)
   except ValueError:continue
   xx=float(t.get('x','0'));yy=float(t.get('y','0'));page=b.global_matrix(t)@np.array([xx,yy,1])
   if page[1]>520 and 0<val<.2 and ((page[0]<180)==(j==0)):
    frac=(val-oldlo)/(oldhi-oldlo);b.settext(t,f'{nlo+frac*(nhi-nlo):.3f}')
  table.to_csv(W/'data'/f'Figure5b_{metric}.csv')
 # Keep the original default-cell outline, 5x6 cells, axes, typography and palette.
 assert sum('fill:none' not in p.get('style','') for p in g.xpath('.//s:path[@clip-path="url(#pc1fa67d900)"]',namespaces=b.NS))==30
def align_cohort_means(r,fx):
 axis_matrix=b.global_matrix(b.ident(r,'axes_8'))
 for cohort,id in [('development','path856'),('external','path857')]:
  p=b.ident(r,id);m=b.global_matrix(p);_,_,y0,y1=b.bbox(p);pts=[]
  for yy in [y0,y1]:
   world=m@np.array([0,yy,1]);world[0]=(axis_matrix@np.array([fx(float(b.maps[b.maps.cohort.eq(cohort)].modified_fraction.mean()*100)),0,1]))[0];pts.append((np.linalg.inv(m)@world)[:2])
  p.set('d',b.poly(pts))
 for cohort,id,anchor,offset in [('development','text1172','start',4),('external','text1203','end',12)]:
  t=b.ident(r,id);m=b.global_matrix(t);target=(axis_matrix@np.array([fx(float(b.maps[b.maps.cohort.eq(cohort)].modified_fraction.mean()*100)),0,1]))[0]+offset;local=np.linalg.inv(m)@np.array([target,0,1]);dx=local[0]-float(t.get('x'));t.set('x',str(local[0]));t.set('style',t.get('style','')+';text-anchor:'+anchor)
  for span in t:
   if span.get('x') is not None:span.set('x',str(float(span.get('x'))+dx))
 b.shift_page(b.ident(r,'g1211'),dy=-8.5)
def restore_original_marker_size(r):
 original=E.parse(str(ORIG/'Figure05_robustness_redesign.svg'));a=b.ident(original,'axes_8');templates={}
 for u in b.uses(a):
  if u.xpath('ancestor::s:g[starts-with(@id,"legend") or @id="PathCollection_5" or @id="PathCollection_6"]',namespaces=b.NS):continue
  for color in ['#2c6eaa','#e66b25']:
   if 'fill:'+color in u.get('style','') and color not in templates:templates[color]=b.global_matrix(u)
 assert len(templates)==2
 parent=b.global_matrix(b.ident(r,'axes_8'));error=0.
 for u in r.xpath('//*[starts-with(@id,"corrected_map_")]'):
  color=next(c for c in templates if 'fill:'+c in u.get('style',''));old=b.global_matrix(u);center=old@np.array([float(u.get('x')),float(u.get('y')),1]);target=old.copy();target[:2,:2]=templates[color][:2,:2];local=np.linalg.inv(parent)@target;u.set('transform',f'matrix({local[0,0]},{local[1,0]},{local[0,1]},{local[1,1]},{local[0,2]},{local[1,2]})');xy=np.linalg.inv(target)@center;b.setuse(u,x=xy[0],y=xy[1]);error=max(error,float(np.abs(b.global_matrix(u)@np.array([float(u.get('x')),float(u.get('y')),1])-center).max()))
 assert error<1e-6
 (W/'qa/ORIGINAL_MARKER_SIZE_VERIFICATION.json').write_text(json.dumps(dict(original_physical_marker_scaling_restored=True,marker_centers_preserved_max_error_pt=error,source_linear_matrices={k:v[:2,:2].tolist() for k,v in templates.items()}),indent=2),encoding='utf-8')
def figure5(with_parameters=True):
 r=E.parse(str(W/'before/figures/Figure5.svg'))
 if with_parameters:restore_parameter_panel(r)
 restore_original_marker_size(r)
 full=copy.deepcopy(r);align_cohort_means(full,lambda v:50.6124+np.asarray(v)/40*523.9872);edge=b.ident(full,'text845');edge.set('style',__import__('re').sub(r'text-anchor\s*:[^;]+','text-anchor:end',edge.get('style','')));full.write(str(W/'figures/Figure5_all322_maps.svg'),encoding='utf-8',xml_declaration=True)
 b.maps.to_csv(W/'data/Figure5d_all322_map_points.csv',index=False)
 a=b.ident(r,'axes_8');axis_matrix=b.global_matrix(a);x0=50.6124;width=523.9872;fx=lambda v:x0+(np.asarray(v)-10)/30*width
 omitted=[]
 for i,row in enumerate(b.maps.itertuples()):
  u=b.ident(r,f'corrected_map_{i:03d}');m=b.global_matrix(u);q=m@np.array([float(u.get('x')),float(u.get('y')),1]);axis_local=np.linalg.inv(axis_matrix)@q;assert abs(axis_local[0]-(x0+row.modified_fraction/0.40*width))<.002
  if row.modified_fraction==0:
   assert row.delta_Mean_Best_IoU==0 and row.accepted_candidates==0;omitted.append(dict(dataset=row.dataset,workflow=row.workflow,cohort=row.cohort,modified_fraction=0,gain=0));u.getparent().remove(u)
  else:
   axis_local[0]=fx(100*row.modified_fraction);q=axis_matrix@axis_local;local=np.linalg.inv(m)@q;b.setuse(u,x=local[0]);assert 10<100*row.modified_fraction<40
 for j in range(5):
  if j==4:
   b.ident(r,'text845').set('display','none')
   for ln in [113,114]:b.ident(r,f'line2d_{ln}').set('display','none')
   continue
  value=10+j*10;t=b.ident(r,f'text{841+j}');b.settext(t,str(value));t.set('x',str(float(fx(value))))
  if value==40:t.set('style',__import__('re').sub(r'text-anchor\s*:[^;]+','text-anchor:end',t.get('style','')))
  for ln in [105+2*j,106+2*j]:
   gg=b.ident(r,f'line2d_{ln}')
   for p in b.paths(gg):
    _,_,y0,y1=b.bbox(p);p.set('d',b.poly([(fx(value),y0),(fx(value),y1)]))
   for u in b.uses(gg):b.setuse(u,x=fx(value))
 # Keep the original labels, typeface and size, aligning them with their
 # recomputed means. Opposite anchors prevent the nearby cohort labels colliding.
 align_cohort_means(r,fx)
 assert len(omitted)==10
 pd.DataFrame(omitted).to_csv(W/'data/Figure5d_zero_edit_maps_outside_zoom.csv',index=False)
 (W/'qa/FIGURE5D_DISPLAY_AUDIT.json').write_text(json.dumps(dict(full_map_count=322,displayed_maps=312,display_xlim=[10,40],outside_zoom=omitted,statistics_use_all_322=True,underlying_rows_removed=False),indent=2),encoding='utf-8')
 r.write(str(W/'figures/Figure5.svg'),encoding='utf-8',xml_declaration=True)
def export(name):
 p=W/'figures'/f'{name}.svg';out=p.with_suffix('.pdf');subprocess.run(['C:/Program Files/Inkscape/bin/inkscape.com',str(p),'--export-type=pdf','--export-filename='+str(out)],check=True,capture_output=True)
 n=3 if name=='Figure3' else 5;size=(2195,2735) if n==3 else (2484,3509)
 def raster(pdf):
  with fitz.open(pdf) as d:
   pix=d[0].get_pixmap(matrix=fitz.Matrix(size[0]/d[0].rect.width,size[1]/d[0].rect.height),alpha=False);im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);return im.resize(size,Image.Resampling.LANCZOS) if im.size!=size else im
 new=raster(out)
 new.save(W/'qa'/f'{name}_full_vector_render.png')
 # Same renderer for original/current SVG, then preserve untouched image pixels.
 bp=W/'qa'/f'Figure{n}_baseline.pdf'
 if not bp.exists():subprocess.run(['C:/Program Files/Inkscape/bin/inkscape.com',str(W/'before/figures'/f'Figure{n}.svg'),'--export-type=pdf','--export-filename='+str(bp)],check=True,capture_output=True)
 old=raster(bp);orig=Image.open(W/'before/figures'/f'Figure{n}.png').convert('RGB');mask=Image.fromarray((np.max(abs(np.asarray(new,np.int16)-np.asarray(old,np.int16)),axis=2)>1).astype('uint8')*255).filter(ImageFilter.MaxFilter(5))
 # Replace the complete changed plotting panels so legacy raster-only tick
 # traces cannot survive in spaces vacated by the updated vector geometry.
 draw=ImageDraw.Draw(mask);sx=size[0]/(526.79952 if n==3 else 595.44);sy=size[1]/(656.39952 if n==3 else 841.68)
 rects=[(0,26,526.79952,235)] if n==3 else [(0,300,345,580),(0,585,595.44,841.68)]
 for x0,y0,x1,y1 in rects:draw.rectangle((round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy)),fill=255)
 final=Image.composite(new,orig,mask);final.save(p.with_suffix('.png'),dpi=(300,300));mask.save(W/'qa'/f'{name}_edit_mask.png')
 print('EXPORTED',name,flush=True)
if __name__=='__main__':
 backup();figure3();export('Figure3')
 if (W/'evaluation/PARAMETER_SENSITIVITY_30.csv').exists():figure5();export('Figure5');export('Figure5_all322_maps')
