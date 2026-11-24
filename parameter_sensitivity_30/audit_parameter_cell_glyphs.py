"""Check the actual PDF glyphs instead of discontiguous merged text spans."""
import sys,json
from pathlib import Path
sys.path.insert(0,'H:/RadarST_FigureRevision2_20261111')
import revise_figures as f
import numpy as np
import fitz
w=f.W;b=f.b
r=f.E.parse(str(w/'figures/Figure5.svg'))
summary=f.pd.read_csv(w/'evaluation/PARAMETER_SENSITIVITY_30.csv')
with fitz.open(w/'figures/Figure5.pdf') as doc:
 page=doc[0];chars=[c for block in page.get_text('rawdict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans'] for c in span['chars'] if c['c'].strip()]
 records=[]
 for metric,clip in [('sample_equal_size_1_10_gain','pc1fa67d900'),('minimum_method_gain','p9a62b427f0')]:
  table=summary.pivot(index='expression_weight',columns='retention_fraction',values=metric).reindex(index=[.90,.85,.80,.75,.70],columns=[.95,.96,.97,.98,.99,1.0]).to_numpy()
  cells=[p for p in r.xpath('.//*[@clip-path=$cl]',cl='url(#'+clip+')') if p.tag==b.S+'path' and 'fill:none' not in p.get('style','')]
  cells.sort(key=lambda p:(round(b.bbox(p)[2],3),b.bbox(p)[0]));assert len(cells)==30
  for k,p in enumerate(cells):
   x0,x1,y0,y1=b.bbox(p);m=b.global_matrix(p);corners=np.array([m@np.array([x,y,1]) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]])
   left,top=corners[:,:2].min(0);right,bottom=corners[:,:2].max(0)
   found=sorted([c for c in chars if left<c['origin'][0]<right and top<c['origin'][1]<bottom],key=lambda c:c['origin'][0]);text=''.join(c['c'] for c in found);expected=f'{table.flat[k]:.3f}';assert text==expected,(metric,k,text,expected)
   margin=min(min(c['bbox'][0]-left,c['bbox'][1]-top,right-c['bbox'][2],bottom-c['bbox'][3]) for c in found);assert margin>1,(metric,k,margin)
   records.append(dict(metric=metric,cell=k,visible_text=text,minimum_glyph_inset_pt=float(margin)))
 result=dict(status='PASS',cells=60,all_visible_numbers_match_data=True,all_glyphs_inside_own_cells=True,minimum_glyph_inset_pt=min(v['minimum_glyph_inset_pt'] for v in records),raw_span_collision_flags_reviewed_as_discontiguous_text_span_artifacts=True,records=records)
 (w/'qa/FIGURE5_GLYPH_LEVEL_HEATMAP_REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
