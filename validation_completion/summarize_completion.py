import os,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,zipfile,shutil,re
from collections import Counter
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from complete_controls import ROOT,SRC,EXT,WORK,REG,sha,put,paths
OLD=Path('G:/RadarST_PROJECT/03_biology/04_cfar_attribution');BIO=ROOT/'biology'
def main():
 regs=pd.read_csv(REG,keep_default_na=False);summary=pd.read_csv(ROOT/'MATCHED_CONTROL_COHORT_SUMMARY.csv');d=pd.read_csv(ROOT/'MATCHED_CONTROL_ALL322_RESULTS.csv');rr=[];rng=np.random.default_rng(20260812)
 for cohort,g in d.groupby('cohort'):
  sec=g.groupby(['control','replicate','dataset']).mean(numeric_only=True).reset_index();m=sec[sec.control.isin(['RadarST','molecular_order'])].pivot(index='dataset',columns='control',values=['Mean_Best_IoU','exact','size3_5_Mean','loss_fraction']);draws=rng.integers(0,len(m),size=(10000,len(m)))
  for metric in ['Mean_Best_IoU','exact','size3_5_Mean','loss_fraction']:
   dd=(m[metric]['RadarST']-m[metric]['molecular_order']).dropna();boots=dd.to_numpy()[rng.integers(0,len(dd),size=(10000,len(dd)))].mean(1);rr.append(dict(cohort=cohort,metric=metric,delta_RadarST_minus_molecular=float(dd.mean()),ci_lower=float(np.quantile(boots,.025)),ci_upper=float(np.quantile(boots,.975)),sections=len(dd),positive_sections=int((dd>1e-12).sum()),negative_sections=int((dd<-1e-12).sum()),tie_sections=int((dd.abs()<=1e-12).sum()),interpretation='paired descriptive section bootstrap; exposed cohorts, no model selection'))
 pd.DataFrame(rr).to_csv(ROOT/'MOLECULAR_ORDER_PAIRED_BOOTSTRAP.csv',index=False)
 # Fix explicit provenance of metadata fields: CFAR median is not an original
 # proposal score. No pathway/random calculation uses this descriptive column.
 meta=pd.read_parquet(BIO/'intermediate/candidate_metadata.parquet');olds=pd.read_parquet(OLD/'intermediate/candidate_metadata.parquet');audit=[]
 for ds,g in meta.groupby('sample'):
  reg=regs[regs.dataset.eq(ds)].iloc[0].to_dict();inp,_,_=paths(reg);pool=pd.read_parquet(inp/'CANDIDATES.parquet');raw=pd.read_parquet(inp/'BASE_FEATURES.parquet');lk={c:i for i,c in enumerate(pool.candidate_id)}
  original=np.array([np.median([raw['raw::proposal_score'].iloc[lk[s]] for s in row.source_candidate_id.split(';')]) for row in g.itertuples()]);meta.loc[g.index,'formal_proposal_score']=original;meta.loc[g.index,'formal_proposal_score_definition']='median original frozen proposal score across identities of the same physical shape'
  with np.load(OLD/'intermediate/samples'/f'{ds}.npz',allow_pickle=False) as z:spots=z['spot_ids'].astype(str);lookup={s:i for i,s in enumerate(spots)};gs=z['gene_signed']
  with np.load(BIO/'intermediate/candidates'/f'{ds}.npz',allow_pickle=False) as z:actual=z['signed'];keys=z['candidate_key'].astype(str)
  assert keys.tolist()==g.candidate_key.tolist()
  check=np.unique(np.linspace(0,len(g)-1,min(20,len(g)),dtype=int));err=0.
  for i in check:
   mem=[lookup[s] for s in g.iloc[i].member_spots.split(';')];expected=np.median(gs[mem],axis=0);err=max(err,float(np.max(np.abs(expected-actual[i]))))
  assert err==0;audit.append(dict(dataset=ds,regions=len(g),independent_gene_median_rows=len(check),max_abs_error=err,old_panel_physical_count=len(olds[olds['sample']==ds].drop_duplicates('member_spots'))))
 meta.to_parquet(BIO/'intermediate/candidate_metadata.parquet',index=False);pd.DataFrame(audit).to_csv(BIO/'INDEPENDENT_GENE_AGGREGATION_QA.csv',index=False)
 # Historical pathology cases: inspect the same shapes and original workflow(s).
 cpath=Path('G:/RadarST_PROJECT/02_formal_pipeline/benchmark/results/manuscript/78A_RadarST_biological_case_studies/78A_selected_case_studies.csv');cases=pd.read_csv(cpath);allcases=[]
 for case in cases.to_dict('records'):
  ds=case['dataset_id'];mem=set(str(case['member_spots']).split(';'))
  for reg in regs[regs.dataset.eq(ds)].to_dict('records'):
   inp,pred,_=paths(reg);a=np.load(inp/'MAP.npz',allow_pickle=False);spots=a['spots'].astype(str);lk={s:i for i,s in enumerate(spots)};ix={lk[s] for s in mem};lab=pd.read_csv(pred/'FINAL_LABELS.csv').refined.to_numpy();edge=a['edge'];same=lab[edge[:,0]]==lab[edge[:,1]];e=edge[same];_,cc=connected_components(csr_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(spots),len(spots))),directed=False);qs=[]
   for c in np.unique(cc[list(ix)]):
    region=set(np.flatnonzero(cc==c));qs.append(len(ix&region)/len(ix|region))
   selected=pd.read_parquet(pred/'SELECTED_OBJECTS.parquet');keys={';'.join(sorted(s.split(';'))) for s in selected.member_spots};key=';'.join(sorted(mem))
   allcases.append(dict(dataset=ds,workflow=reg['workflow'],case_type=case['case_type'],original_candidate_key=case['candidate_key'],original_case_workflows=case['methods'],is_original_workflow=reg['workflow'] in str(case['methods']).split(';'),member_spots=key,original_shape_current_predicted_best_IoU=max(qs),original_shape_is_complete_predicted_instance=max(qs)>=1-1e-12,original_shape_selected=key in keys,endpoint='shape identity audit, not pathology-reference best-IoU',selection_scope='pre-existing fixed cases; no new case chosen by endpoint'))
 pd.DataFrame(allcases).to_csv(BIO/'HISTORICAL_PATHOLOGY_CASE_ALL_WORKFLOW_AUDIT.csv',index=False)
 stats=json.loads((BIO/'BIOLOGY_COMPLETION_RECEIPT.json').read_text());rand=pd.read_csv(BIO/'cfar_random_calibration.csv');cf=rand[rand.metric.eq('CFAR intensity')];stats.update(random_positive_cells=int((cf.observed_minus_random_median>0).sum()),random_total_cells=len(cf),median_random_cfar_difference=float(cf.observed_minus_random_median.median()),old_physical_panel_size=int(olds.drop_duplicates(['sample','member_spots']).shape[0]),independent_gene_aggregation_rows=sum(r['independent_gene_median_rows'] for r in audit),independent_gene_max_error=max(r['max_abs_error'] for r in audit));put(BIO/'BIOLOGY_COMPLETION_RECEIPT.json',stats)
 evidence={str(p):sha(p) for p in [REG,WORK/'PAPER_ALL_COMPONENT_EVENTS.csv',WORK/'PAPER_ALL322_MAP_METRICS.csv',cpath,OLD/'CFAR_BIOLOGY_LOCK.json',OLD/'intermediate/REPRODUCTION_GATE.json',OLD/'scripts/cfar_biological_attribution.py']+list((OLD/'inputs/gene_sets').glob('*.gmt'))};put(ROOT/'audits/ADDITIONAL_INPUT_HASHES.json',evidence)
 # Protocol deviations are engineering handling only; failures remain in logs.
 put(ROOT/'audits/ENGINEERING_FAILURES_AND_CORRECTIONS.json',dict(preserved_logs=['biology_census.log','biology_census_resume1.log','biology_complete_resume2.log'],corrections=['Unassigned and source labels outside frozen focal-context mapping kept unassigned with original label, never normal tissue','Use original nanmedian for spot-fidelity arrays; undefined spot correlation is not zero or an excluded section','Metadata original proposal score distinguished from median spot CFAR; this did not change any pathway or control result'],model_or_threshold_changes=False))
 print('COMPLETION_STATISTICS_AND_CASES_VERIFIED',stats,flush=True)
if __name__=='__main__':main()
