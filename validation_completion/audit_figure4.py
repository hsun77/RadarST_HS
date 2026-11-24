import sys
sys.dont_write_bytecode=True
from complete_controls import ROOT,WORK,sha,put
from pathlib import Path
import pandas as pd,hashlib
SOURCE=Path('G:/RadarST_PROJECT/01_manuscript/figure_source_data/Figure04_rescue_redesign')
e=pd.read_csv(SOURCE/'Figure_4_panel_a_examples.csv');m=pd.read_csv(SOURCE/'Figure_4_panel_a_component_membership.csv',dtype=str);t=pd.read_csv(WORK/'PAPER_ALL_COMPONENT_EVENTS.csv');rows=[]
for c in e.itertuples():
 members=sorted(m.loc[(m.dataset_id==c.dataset_id)&m.annotated_component_id.eq(c.annotated_component_id),'spot_id']);assert len(members)==c.component_size_spots
 key=hashlib.sha256((c.dataset_id+'|'+';'.join(members)).encode()).hexdigest();masks=t.dataset_id.eq(c.dataset_id)&t.truth_component_id.eq(key);m2=t.loc[masks];assert len(m2)==14,(c.dataset_id,c.annotated_component_id,key,len(m2))
 for x in m2.itertuples():
  rows.append(dict(cohort=x.evaluation_cohort,dataset=x.dataset_id,workflow=x.method,source_case_id=c.annotated_component_id,truth_component_id=key,truth_label=x.truth_label,component_size=x.component_size_spots,reference_members=x.reference_members,upstream_best_IoU=x.base_best_IoU,RadarST_best_IoU=x.after_best_IoU,delta=x.delta_best_IoU,exact=x.after_best_IoU>=1-1e-12,near_0_8=x.after_best_IoU>=.8-1e-12,is_original_display_workflow=x.method==c.method,original_displayed_best_IoU=c.after_best_IoU,case_selection='original fixed nine cases; no reselection',reference_membership_match='EXACT'))
d=pd.DataFrame(rows);assert len(e)==9 and len(d)==126;d.to_csv(ROOT/'biology/FIGURE4_NINE_CASES_ALL_WORKFLOWS.csv',index=False);d[d.is_original_display_workflow].to_csv(ROOT/'biology/FIGURE4_NINE_DISPLAYED_CASES_RECOMPUTED.csv',index=False)
put(ROOT/'audits/FIGURE4_CASE_SOURCE_HASHES.json',{str(p):sha(p) for p in [SOURCE/'Figure_4_panel_a_examples.csv',SOURCE/'Figure_4_panel_a_component_membership.csv']})
print(d[d.is_original_display_workflow][['dataset','source_case_id','upstream_best_IoU','RadarST_best_IoU','original_displayed_best_IoU']].to_string(index=False))
