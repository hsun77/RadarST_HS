"""Independent data/geometry checks for the two revised figures."""
import sys,json,re,hashlib
sys.path.insert(0,'H:/RadarST_FigureRevision2_20261111')
import revise_figures as f
import numpy as np
from PIL import Image
W=f.W;E=f.E;b=f.b
def scalar_style(style):return re.sub(r'(?:^|;)\s*fill\s*:[^;]*','',style or '')
def main():
 records={}
 for n,axes in [(3,[3,4,5,6]),(5,[1,2,7])]:
  before=E.parse(str(W/'before/figures'/f'Figure{n}.svg'));after=E.parse(str(W/'figures'/f'Figure{n}.svg'))
  for key in ['width','height','viewBox']:assert before.getroot().get(key)==after.getroot().get(key)
  identical={}
  for k in axes:
   old=b.ident(before,f'axes_{k}');new=b.ident(after,f'axes_{k}')
   if n==3 and k==4:
    import copy
    new=copy.deepcopy(new);t=b.ident(new,'text_90');prior=b.ident(old,'text_90')
    if prior.get('transform') is None:t.attrib.pop('transform',None)
    else:t.set('transform',prior.get('transform'))
   identical[str(k)]=E.tostring(old)==E.tostring(new)
  assert all(identical.values()),identical
  source=np.asarray(Image.open(W/'before/figures'/f'Figure{n}.png').convert('RGB'));result=np.asarray(Image.open(W/'figures'/f'Figure{n}.png').convert('RGB'));mask=np.asarray(Image.open(W/'qa'/f'Figure{n}_edit_mask.png'))>0;assert np.array_equal(source[~mask],result[~mask])
  records[f'Figure{n}']=dict(unchanged_axes_XML_byte_identical_except_listed_label_adjustment=identical,listed_label_adjustment='Panel d letter moved upward 5 pt to clear the 100 tick label.' if n==3 else None,canvas_dimensions_unchanged=True,pixels_outside_changed_regions_exactly_preserved=True)
 # Original 5b has identical node geometry, fonts, positions and constraints.
 original=E.parse(str(f.ORIG/'Figure05_robustness_redesign.svg'));revised=E.parse(str(W/'figures/Figure5.svg'));a=list(b.ident(original,'g1201').iter());c=list(b.ident(revised,'g1201').iter());assert len(a)==len(c)
 for old,new in zip(a,c):
  assert old.tag==new.tag
  if not isinstance(old.tag,str):continue
  assert {k:v for k,v in old.attrib.items() if k!='style'}=={k:v for k,v in new.attrib.items() if k!='style'},old.get('id')
  assert scalar_style(old.get('style'))==scalar_style(new.get('style')),old.get('id')
 summary=f.pd.read_csv(W/'evaluation/PARAMETER_SENSITIVITY_30.csv')
 for column,start in [('sample_equal_size_1_10_gain',57),('minimum_method_gain',95)]:
  expected=summary.pivot(index='expression_weight',columns='retention_fraction',values=column).reindex(index=[.90,.85,.80,.75,.70],columns=[.95,.96,.97,.98,.99,1.]).to_numpy().flat
  for i,val in enumerate(expected):assert ''.join(b.texts(b.ident(revised,f'text_{start+i}'))[0].itertext()).strip()==f'{val:.3f}'
 records['Figure5b']=dict(original_node_geometry_fonts_positions_preserved=True,actual_configurations=30,heatmaps=2,both_cell_value_arrays_match_real_results=True,locked_reported_cell_preserved=True)
 axis_matrix=b.global_matrix(b.ident(revised,'axes_8'));points=0;err=0.
 for i,row in enumerate(b.maps.itertuples()):
  hits=revised.xpath('//*[@id=$id]',id=f'corrected_map_{i:03d}')
  if row.modified_fraction==0:assert not hits;continue
  assert len(hits)==1;u=hits[0];v=np.linalg.inv(axis_matrix)@b.global_matrix(u)@np.array([float(u.get('x')),float(u.get('y')),1]);area=10+(v[0]-50.6124)/523.9872*30;gain=.55-(v[1]-(634.94818+18.5467))/152.02261*.83
  err=max(err,abs(area/100-row.modified_fraction),abs(gain-row.delta_Mean_Best_IoU));points+=1
 assert points==312 and err<1e-8
 for cohort,id in [('development','path856'),('external','path857')]:
  p=b.ident(revised,id);x,_,y,_=b.bbox(p);v=np.linalg.inv(axis_matrix)@b.global_matrix(p)@np.array([x,y,1]);area=10+(v[0]-50.6124)/523.9872*30;assert abs(area/100-b.maps[b.maps.cohort.eq(cohort)].modified_fraction.mean())<1e-8
 records['Figure5d']=dict(displayed_points=points,all_statistical_rows=322,outside_zoom_zero_edit_maps=10,recovered_data_max_error=err,cohort_mean_lines_match_all_322_maps=True)
 (W/'qa/FINAL_DATA_AND_STYLE_VERIFICATION.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(records))
if __name__=='__main__':main()
