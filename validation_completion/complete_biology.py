"""Reaggregate fixed CFAR arrays for actual final selected physical objects."""
import os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'biology'
os.environ['MPLCONFIGDIR']=str(ROOT/'logs/matplotlib');sys.dont_write_bytecode=True
import json,hashlib,ast,re,importlib.util,shutil,time
from collections import defaultdict,Counter
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
SRC=Path('H:/RadarST_2_ObjectValue_20261104');EXT=Path('H:/RadarST_E2_FixedTransfer_20261105')
OLD=Path('G:/RadarST_PROJECT/03_biology/04_cfar_attribution')
sys.path.insert(0,str(Path('G:/RadarST_PROJECT/03_biology/01_local_expression/vendor')))
sys.path.insert(0,str(ROOT/'code'))
from complete_controls import sha,put,paths,REG
def protect(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  mode,flags=args[1:3];wr=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
  if wr:
   p=Path(os.fsdecode(args[0])).resolve()
   if any(p.is_relative_to(q) for q in [Path('G:/'),SRC,EXT,Path('H:/RadarST_PROJECT/NC_SUBMISSION_FINAL')]):raise RuntimeError('PROTECTED_WRITE '+str(p))
sys.addaudithook(protect)
def vendor():
 target=ROOT/'code/original_cfar_biological_attribution.py'
 if not target.exists():shutil.copyfile(OLD/'scripts/cfar_biological_attribution.py',target)
 assert sha(target)==sha(OLD/'scripts/cfar_biological_attribution.py')
 sp=importlib.util.spec_from_file_location('frozen_cfar_attribution',target);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);m.PROJECT=OUT
 m.GENESETS={k:OLD/'inputs/gene_sets'/p.name for k,p in m.GENESETS.items()}
 return m
def census(m):
 regs=pd.read_csv(REG,keep_default_na=False);old=pd.read_parquet(OLD/'intermediate/candidate_metadata.parquet')
 oldkeys={(r.sample,';'.join(sorted(r.member_spots.split(';')))):r for r in old.itertuples()}
 p=Path('G:/RadarST_PROJECT/02_formal_pipeline/benchmark/scripts/78D_RadarST_pathology_stratified_biological_context.py');fn=next(x for x in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='biological_context');ns={'re':re};exec(compile(ast.Module([fn],type_ignores=[]),str(p),'exec'),ns)
 allrows=[];provenance={};audits=[];instances=[];current={};case_rows=[]
 for ds,g in regs.groupby('dataset',sort=True):
  t=time.time();reg=g.iloc[0].to_dict();inp,_,_=paths(reg);a=np.load(inp/'MAP.npz',allow_pickle=False);spots=a['spots'].astype(str);idx={s:i for i,s in enumerate(spots)}
  f=OLD/'intermediate/samples'/f'{ds}.npz';provenance[str(f)]=sha(f)
  with np.load(f,allow_pickle=False) as z:
   need=['spot_ids','genes','coords','cfar','gene_signed','gene_magnitude','gene_normalized','contribution','fidelity_rho'];z={k:z[k] for k in need}
  assert list(z['spot_ids'].astype(str))==list(spots),(ds,'SPOT_ALIGNMENT')
  pool=pd.read_parquet(inp/'CANDIDATES.parquet');raw=pd.read_parquet(inp/'BASE_FEATURES.parquet');sing=pool.n_spots.eq(1);sv=np.array([z['cfar'][idx[s]] for s in pool.loc[sing,'member_spots']]);err=float(np.max(np.abs(raw.loc[sing,'raw::proposal_score'].to_numpy()-sv)))
  assert err<=1e-8,(ds,'CFAR_FRONTEND_PARITY',err)
  shapes={};memberships={};cx={}
  truth=pd.read_csv(reg['reference'],dtype=str,keep_default_na=False).set_index('spot_id').truth_label
  for r in g.to_dict('records'):
   ip,pr,_=paths(r);assert sha(ip/'CANDIDATES.parquet')==sha(inp/'CANDIDATES.parquet'),ds
   sel=pd.read_parquet(pr/'SELECTED_OBJECTS.parquet');labels=pd.read_csv(pr/'FINAL_LABELS.csv',dtype={'spot_id':str});assert labels.spot_id.tolist()==spots.tolist();used=set()
   for row in sel.itertuples():
    mem=sorted(row.member_spots.split(';'));key=';'.join(mem);loc=[idx[s] for s in mem];assert not set(loc)&used;used.update(loc)
    vals=labels.refined.iloc[loc].to_numpy();assert np.unique(vals).size==1;assert int((labels.refined==vals[0]).sum())==len(mem)
    q=shapes.setdefault(key,dict(wf=[],ids=set(),types=set()));q['wf'].append(r['workflow']);q['ids'].add(row.candidate_id);q['types'].add(row.proposal_type)
    instances.append(dict(cohort=r['cohort'],dataset=ds,workflow=r['workflow'],candidate_id=row.candidate_id,member_spots=key,final_identity='EXACT_MEMBER_MATCH',n_spots=len(mem)))
   provenance[str(pr/'FINAL_LABELS.csv')]=sha(pr/'FINAL_LABELS.csv');provenance[str(pr/'SELECTED_OBJECTS.parquet')]=sha(pr/'SELECTED_OBJECTS.parquet')
  rows=[];signed=[];mag=[];norm=[];pc=[];tree=cKDTree(z['coords']);cfar=z['cfar']
  for key,q in sorted(shapes.items()):
   loc=np.array([idx[s] for s in key.split(';')],int);cid=hashlib.sha256((ds+'|'+key).encode()).hexdigest();common=oldkeys.get((ds,key));center=z['coords'][loc].mean(0);_,order=tree.query(center,k=min(len(spots),len(loc)+60));bg=[int(i) for i in np.atleast_1d(order) if i not in loc][:30]
   labels=truth.reindex(key.split(';'));assert labels.notna().all();cnt=Counter(labels);lab=sorted(cnt,key=lambda k:(-cnt[k],k))[0]
   # Unknown/excluded labels remain unassigned, never benign tissue.
   try:context=ns['biological_context'](lab)
   except ValueError:context='unassigned' # preserve unmapped source label below
   c=dict(sample=ds,candidate_key=ds+'||'+cid,candidate_id=cid,source_candidate_id=';'.join(sorted(q['ids'])),proposal_type=';'.join(sorted(q['types'])),member_spots=key,background_spots=';'.join(spots[bg]),n_background=len(bg),candidate_size=len(loc),cohort=reg['cohort'],cancer_type=m.SAMPLE_SPECS[ds][1],biological_context=context,workflow_recurrence_count=len(q['wf']),formal_workflows_supporting_region=';'.join(sorted(q['wf'])),candidate_molecular_score=float(np.median(cfar[loc])),formal_proposal_score=float(np.median(cfar[loc])),mean_spot_cfar=float(np.mean(cfar[loc])),median_spot_cfar=float(np.median(cfar[loc])),max_spot_cfar=float(np.max(cfar[loc])),median_member_spot_attribution_fidelity=float(np.nanmedian(z['fidelity_rho'][loc])),section_median_spot_attribution_fidelity=float(np.nanmedian(z['fidelity_rho'])),gene_index_id=ds,in_historical_panel=common is not None,context_rule='majority frozen spot reference label; lexical tie; descriptive only',background_rule='30 nearest centroid spots outside object')
   c['majority_source_label']=lab;c['context_mapping_status']='UNMAPPED_OR_EXCLUDED' if context=='unassigned' else 'FROZEN_MAPPING'
   rows.append(c);signed.append(np.median(z['gene_signed'][loc],axis=0));mag.append(np.median(z['gene_magnitude'][loc],axis=0));norm.append(np.median(z['gene_normalized'][loc],axis=0));pc.append(np.mean(z['contribution'][loc],axis=0))
  frame=pd.DataFrame(rows);allrows.extend(rows);current[ds]=shapes
  np.savez_compressed(OUT/'intermediate/candidates'/f'{ds}.npz',candidate_key=frame.candidate_key.to_numpy(dtype='U'),genes=z['genes'],signed=np.asarray(signed,np.float32),magnitude=np.asarray(mag,np.float32),normalized=np.asarray(norm,np.float32),pc_share=np.asarray(pc,np.float32))
  audits.append(dict(dataset=ds,objects=len(frame),historical_overlap=int(frame.in_historical_panel.sum()),new_final_shapes=int((~frame.in_historical_panel).sum()),singleton_cfar_max_abs_diff=err,all_spots_aligned=True,all_final_instances_exact=True,section_spot_fidelity=float(np.nanmedian(z['fidelity_rho'])),seconds=time.time()-t));print('BIO_CENSUS',ds,len(frame),audits[-1]['new_final_shapes'],flush=True)
 pd.DataFrame(allrows).to_parquet(OUT/'intermediate/candidate_metadata.parquet',index=False);pd.DataFrame(audits).to_csv(OUT/'CURRENT_PANEL_CENSUS_AND_FRONTEND_PARITY.csv',index=False);pd.DataFrame(instances).to_parquet(OUT/'ALL322_SELECTED_INSTANCE_IDENTITIES.parquet',index=False)
 fidelity=np.array([v['section_spot_fidelity'] for v in audits]);passed=float(np.median(fidelity))>.1 and float(np.mean(fidelity>0))>=.8 and float(fidelity.min())>=-.05;assert passed
 for lockname in ['PANEL_D_CASE_LOCK.json','REPRESENTATIVE_CONTEXT_CASE_LOCK.json','REPRESENTATIVE_CASE_LOCK.json']:
  payload=json.loads((OLD/lockname).read_text());cases=payload.get('selected_candidates',payload.get('cases',[]))
  if isinstance(cases,dict):cases=list(cases.values())
  if not cases and 'sample' in payload:cases=[payload]
  for case in cases:
   ds=case.get('sample','');members=case.get('member_spots','')
   if not members:
    key=case.get('candidate_key','');hit=old[old.candidate_key==key]
    if len(hit)==1:members=hit.iloc[0].member_spots
   if not ds or not members:case_rows.append(dict(lock=lockname,status='UNRESOLVED_RECORD',payload=json.dumps(case)));continue
   key=';'.join(sorted(members.split(';')));v=current.get(ds,{}).get(key);case_rows.append(dict(lock=lockname,sample=ds,member_spots=key,selected_workflows=';'.join(sorted(v['wf'])) if v else '',selected_workflow_count=len(v['wf']) if v else 0,status='RETAINED_IN_FINAL_MAPS' if v else 'NOT_A_FINAL_SELECTED_OBJECT',histology=case.get('histology_source','')))
 pd.DataFrame(case_rows).to_csv(OUT/'HISTORICAL_BIOLOGICAL_CASE_MEMBERSHIP_AUDIT.csv',index=False)
 put(OUT/'INPUT_PROVENANCE.json',provenance);put(OUT/'intermediate/REPRODUCTION_GATE.json',dict(reproduction_pass=True,attribution_unlocked=True,maximum_singleton_cfar_abs_error=max(x['singleton_cfar_max_abs_diff'] for x in audits),n_sections=len(audits),n_canonical_regions=len(allrows),fidelity_pass=passed,fidelity_median_section=float(np.median(fidelity)),original_spot_archives_reused=True,no_model_change=True))
 return pd.DataFrame(allrows)
def main():
 start=time.time()
 for sub in ['intermediate/candidates','intermediate/pathways']: (OUT/sub).mkdir(parents=True,exist_ok=True)
 m=vendor();meta=pd.read_parquet(OUT/'intermediate/candidate_metadata.parquet') if (OUT/'intermediate/REPRODUCTION_GATE.json').exists() else census(m)
 if '--census-only' in sys.argv:return
 m.N_GSEA_PERM=1000;m.N_BLOCK_PERM=10000;m.N_RANDOM=500
 # Four native GSEA workers leaves four for independent map checks.
 m.os.cpu_count=lambda:4
 section,landscape=m.section_gene_and_landscape(meta);print('BIO_GENE_LANDSCAPE_COMPLETE',len(landscape),flush=True)
 m.candidate_explanations(meta)
 libraries,aggregate=m.aggregate_pathways(section,landscape)
 selected=m.select_followup_terms(aggregate,libraries,set(landscape.loc[landscape.eligible_for_pathway,'gene'].astype(str)))
 correlations,payload,context,cancer=m.candidate_pathway_stage(meta,selected,libraries);m.summarize_pathway_associations(correlations,payload,selected);final=m.finalize_pathway_classes(aggregate,context,cancer)
 random=m.random_calibration(meta,selected,libraries)
 # Correct textual graph label while preserving the original numerical code:
 # kneighbors(X=None) excludes self, returning k+1=7 actual nonself neighbors.
 random['random_framework']='symmetrized 7-nonself-neighbor graph; original k=6 implementation; 500 connected sets; base seed 20260805';random.to_csv(OUT/'cfar_random_calibration.csv',index=False)
 put(OUT/'BIOLOGY_COMPLETION_RECEIPT.json',dict(status='COMPLETED',sections=int(meta['sample'].nunique()),physical_objects=len(meta),historical_overlap=int(meta.in_historical_panel.sum()),new_final_shapes=int((~meta.in_historical_panel).sum()),pathway_counts=final.pathway_class.value_counts().to_dict(),candidate_followup_pathways=len(selected),detector_random_cells=int((random.metric=='CFAR intensity').sum()),seconds=time.time()-start,independent_biological_validation=False,scope='descriptive final-object molecular attribution; labels for context only; no figure redraw'))
 print('BIOLOGY_COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
