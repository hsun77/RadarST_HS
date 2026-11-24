from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,gzip,json,math,os,sys,time
import joblib,numpy as np,pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from . import core
from .pipeline import PROJECT,manifests,path,context,truth,base,sha,dump,seal,trusted,summarize,guard

def context_mapping():
    table=pd.read_csv(PROJECT/'configs/truth_context_map.csv')
    return {(r.study_stratum.lower(),r.truth_label):r.biological_context for r in table.itertuples()}

def run_controls(run):
    import traceback
    errors=[]
    for fn in [randomization,sensitivity,context_summaries,external_lopo]:
        started=time.time()
        try:fn(run);status='PASS';error=''
        except Exception:error=traceback.format_exc();status='BLOCKED';errors.append({'name':fn.__name__,'error':error});print(error,flush=True)
        dump(run/'logs'/('experiment_'+fn.__name__+'.json'),{'status':status,'seconds':time.time()-started,'error':error})
    if errors:raise RuntimeError('Independent control experiments failed: '+repr(errors))

def random_worker(task):
    runstr,d,m=task;run=Path(runstr);guard(run,'controls')
    from .legacy import randomization as rand,fast_metrics
    f=core.functions();rand.H=f.r
    rand.H.connected_match_state=fast_metrics.connected_match_state;rand.H.best_match_fast=fast_metrics.best_match_fast
    samples,pairs=manifests();sample=next(s for s in samples.itertuples() if s.dataset==d);row=next(s for s in pairs.itertuples() if s.dataset==d and s.method==m)
    folder=path(run,d,m)/'randomization';featurepath=path(run,d,m)/'features/features.parquet';finalpath=path(run,d,m)/'inference/selected_regions.csv'
    inp={'features':sha(featurepath),'retained':sha(finalpath),'repetitions':1000,'seed':20260812}
    if trusted(folder,inp):return {'dataset':d,'method':m,'status':'TRUSTED_CURRENT_RUN'}
    folder.mkdir(parents=True,exist_ok=True);ctx=context(sample);frame=pd.read_parquet(featurepath);final=pd.read_csv(finalpath);original=base(row,ctx)
    sampler=rand.exact_sampler(frame,final,ctx['spot_ids']);comps=core.components(truth(sample),ctx,sample.source);idx={s:i for i,s in enumerate(ctx['spot_ids'])}
    edges=[(i,j) for i,neighbors in enumerate(ctx['adjacency']) for j in neighbors if i<j]
    ei=np.array([e[0] for e in edges],dtype=np.int32);ej=np.array([e[1] for e in edges],dtype=np.int32)
    codes=pd.factorize(original,sort=True)[0].astype(np.int32);cc,sz=fast_metrics.connected_match_state(codes,ei,ej)
    truth_idx={r.truth_component_id:np.array([idx[s] for s in r.truth_spots],dtype=np.int32) for r in comps.itertuples() if r.component_size_spots<=10}
    matches={k:fast_metrics.best_match_fast(cc,sz,v) for k,v in truth_idx.items()}
    state={'base_codes':codes,'edge_i':ei,'edge_j':ej,'truth_indices':truth_idx}
    rng=np.random.default_rng(rand.pair_seed(d,m,20260812));gains=[];audits=[]
    hist=Counter((r.channel,r.proposal_type,int(r.n_spots)) for r in final.itertuples())
    with threadpool_limits(limits=1):
        for b in range(1000):
            records,burden=rand.draw_records(sampler,rng)
            assert len(records)==len(final) and burden==int(final.n_spots.sum())
            assert len(set().union(*(r['members'] for r in records)))==burden
            assert Counter((token[0],r['proposal_type'],r['n_spots']) for token,r in zip(sampler['tokens'],records))==hist
            gains.append(rand.random_small_gain(state,matches,list(truth_idx),records))
            audits.append({'realization':b,'candidate_count':len(records),'modified_spots':burden,'composition_exact':True,'nonoverlap':True,'legality_masks_verified':True})
    np.save(folder/'gains.npy',np.asarray(gains));pd.DataFrame(audits).to_csv(folder/'MATCHING_AUDIT.csv',index=False);seal(folder,inp)
    return {'dataset':d,'method':m,'status':'FRESH','repetitions':len(gains)}

def randomization(run):
    samples,pairs=manifests();tasks=[(str(run),r.dataset,r.method) for r in pairs[pairs.cohort.ne('control')].itertuples()]
    out=run/'experiments/randomization';out.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        statuses=[]
        for result in pool.map(random_worker,tasks):statuses.append(result);print('RANDOMIZED',result['dataset'],result['method'],flush=True)
    pd.DataFrame(statuses).to_csv(out/'PAIR_STATUS.csv',index=False)
    rows=[]
    for cohort in ['development','external']:
        null=[]
        for d in samples[samples.cohort.eq(cohort)].dataset:
            arrays=[np.load(path(run,d,r.method)/'randomization/gains.npy') for r in pairs[pairs.dataset.eq(d)].itertuples()]
            null.append(np.mean(arrays,axis=0))
        whole=np.mean(null,axis=0);observed=summarize(pd.read_csv(run/'evaluation'/(cohort+'_events.csv.gz')))[0]['gain']
        pd.DataFrame({'realization':range(1000),'sample_equal_gain':whole}).to_csv(out/(cohort+'_whole_benchmark_null.csv'),index=False)
        rows.append({'cohort':cohort,'observed_gain':observed,'random_mean':whole.mean(),'random_q025':np.quantile(whole,.025),'random_q975':np.quantile(whole,.975),'empirical_p_greater':(1+np.sum(whole>=observed))/1001,'repetitions':1000})
    pd.DataFrame(rows).to_csv(out/'RANDOMIZATION_RESULTS.csv',index=False)

def sensitivity(run):
    samples,pairs=manifests();out=run/'experiments/sensitivity';out.mkdir(parents=True,exist_ok=True)
    tasks=[(str(run),r.dataset,r.method) for r in pairs[pairs.cohort.eq('development')].itertuples()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(sensitivity_pair,tasks):print('SENSITIVITY',result,flush=True)
    results=[];audits=[]
    for _,d,m in tasks:
        folder=out/'pairs'/d/m;results.append(pd.read_csv(folder/'events.csv.gz'));audits.append(pd.read_csv(folder/'audit.csv'))
    events=pd.concat(results,ignore_index=True);events.to_csv(out/'events.csv.gz',index=False,compression='gzip');audit=pd.concat(audits,ignore_index=True);audit.to_csv(out/'PAIR_AUDIT.csv',index=False)
    summary=[]
    for (w,r),group in events.groupby(['expression_weight','retention']):
        values,pair,section=summarize(group,seed=20260812);matching=audit[audit.expression_weight.eq(w)&audit.retention.eq(r)]
        summary.append({'expression_weight':w,'retention':r,'minimum_workflow_gain':pair.groupby('method').gain.mean().min(),'minimum_section_gain':section.gain.min(),'median_pair_ARI_delta':matching.ARI_delta.median(),**values})
    pd.DataFrame(summary).to_csv(out/'GRID_RESULTS.csv',index=False)

def sensitivity_pair(task):
    root,d,m=task;run=Path(root);guard(run,'controls');samples,pairs=manifests();sample=next(s for s in samples.itertuples() if s.dataset==d);row=next(r for r in pairs.itertuples() if r.dataset==d and r.method==m)
    out=run/'experiments/sensitivity/pairs'/d/m;out.mkdir(parents=True,exist_ok=True)
    inp={'features':sha(path(run,d,m)/'features/features.parquet'),'ranking':sha(path(run,d,m)/'inference/ranking.parquet'),'slots':sha(path(run,d,m)/'inference/selector_slots.csv'),'weights':[.70,.75,.80,.85,.90],'retention':[.95,.96,.97,.98,.99,1.]}
    if trusted(out,inp):return {'dataset':d,'method':m,'status':'CURRENT_RUN'}
    with threadpool_limits(limits=1):
        f=core.functions();cfg=json.loads((PROJECT/'configs/formal.json').read_text());ctx=context(sample);ref=truth(sample);comps=core.components(ref,ctx,sample.source);original=base(row,ctx)
        frame=pd.read_parquet(path(run,d,m)/'features/features.parquet');ranking=pd.read_parquet(path(run,d,m)/'inference/ranking.parquet');slots=pd.read_csv(path(run,d,m)/'inference/selector_slots.csv');results=[];audits=[]
        for weight in [.70,.75,.80,.85,.90]:
            score=weight*frame.score_per_spot_percentile.to_numpy()+(1-weight)*ranking.score_expected_gain_rank.to_numpy()-.025*frame.cross_component_risk.to_numpy()
            _,selected,search=f.sel.exact_token_assignment(frame,score,score,slots,ctx['spot_ids'],m,cfg['selector_variants']['development'])
            for retention in [.95,.96,.97,.98,.99,1.]:
                budget=math.floor(retention*selected.n_spots.sum()+1e-12);touched=0;labels=original.copy();n=0
                for rr in selected.sort_values(['selection_score','candidate_key'],ascending=[False,True]).itertuples():
                    if touched+rr.n_spots<=budget:labels.loc[rr.member_spots.split(';')]='sensitivity_'+str(n);n+=1;touched+=rr.n_spots
                ev,macro=core.evaluate(original,labels,ref,ctx,comps,m,'development');ev['expression_weight']=weight;ev['retention']=retention;results.append(ev)
                audits.append({'dataset':d,'method':m,'expression_weight':weight,'expected_gain_weight':1-weight,'retention':retention,'modified_spots':touched,'configuration_selected_from_this_grid':False,**macro,**search})
        pd.concat(results,ignore_index=True).to_csv(out/'events.csv.gz',index=False,compression='gzip');pd.DataFrame(audits).to_csv(out/'audit.csv',index=False);seal(out,inp)
    return {'dataset':d,'method':m,'status':'FRESH'}

def sensitivity_serial_reference(run):
    f=core.functions();samples,pairs=manifests();cfg=json.loads((PROJECT/'configs/formal.json').read_text());out=run/'experiments/sensitivity';out.mkdir(parents=True,exist_ok=True)
    results=[];audit=[]
    for sample in samples[samples.cohort.eq('development')].itertuples():
        ctx=context(sample);ref=truth(sample);comps=core.components(ref,ctx,sample.source)
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            original=base(row,ctx);frame=pd.read_parquet(path(run,row.dataset,row.method)/'features/features.parquet')
            ranking=pd.read_parquet(path(run,row.dataset,row.method)/'inference/ranking.parquet');slots=pd.read_csv(path(run,row.dataset,row.method)/'inference/selector_slots.csv')
            for weight in [.70,.75,.80,.85,.90]:
                score=weight*frame.score_per_spot_percentile.to_numpy()+(1-weight)*ranking.score_expected_gain_rank.to_numpy()-.025*frame.cross_component_risk.to_numpy()
                _,selected,search=f.sel.exact_token_assignment(frame,score,score,slots,ctx['spot_ids'],row.method,cfg['selector_variants']['development'])
                for retention in [.95,.96,.97,.98,.99,1.0]:
                    budget=math.floor(retention*selected.n_spots.sum()+1e-12);touched=0;labels=original.copy();n=0
                    for rr in selected.sort_values(['selection_score','candidate_key'],ascending=[False,True]).itertuples():
                        if touched+rr.n_spots<=budget:
                            labels.loc[rr.member_spots.split(';')]='sensitivity_'+str(n);n+=1;touched+=rr.n_spots
                    ev,macro=core.evaluate(original,labels,ref,ctx,comps,row.method,'development');ev['expression_weight']=weight;ev['retention']=retention;results.append(ev)
                    audit.append({'dataset':row.dataset,'method':row.method,'expression_weight':weight,'expected_gain_weight':1-weight,'retention':retention,'modified_spots':touched,'configuration_selected_from_this_grid':False,**search})
            print('SENSITIVITY',row.dataset,row.method,flush=True)
    events=pd.concat(results,ignore_index=True);events.to_csv(out/'events.csv.gz',index=False,compression='gzip');pd.DataFrame(audit).to_csv(out/'PAIR_AUDIT.csv',index=False)
    summary=[]
    for (w,r),group in events.groupby(['expression_weight','retention']):
        values,_,_=summarize(group);summary.append({'expression_weight':w,'retention':r,**values})
    pd.DataFrame(summary).to_csv(out/'GRID_RESULTS.csv',index=False)

def context_summaries(run):
    cmap=context_mapping();out=run/'experiments/pathology';out.mkdir(parents=True,exist_ok=True);rows=[]
    for cohort in ['development','external']:
        events=pd.read_csv(run/'evaluation'/(cohort+'_events.csv.gz'));stratum='development' if cohort=='development' else 'held-out external'
        events['context']=[cmap.get((stratum,t),'unassigned') for t in events.truth_label]
        events.to_csv(out/(cohort+'_context_events.csv.gz'),index=False,compression='gzip')
        for scope,eligible in [('size_1_10',events[events.truth_size.between(1,10)]),('all_components',events)]:
            for label,g in eligible[eligible.context.ne('unassigned')].groupby('context'):
                metrics=['before','after','gain','coverage_before','coverage_after','purity_before','purity_after','dilution_before','dilution_after']
                pair=g.groupby(['dataset','method'])[metrics].mean();section=pair.groupby('dataset').mean();values=section.gain.to_numpy()
                seed=int.from_bytes(__import__('hashlib').sha256((cohort+'|'+label).encode()).digest()[:4],'little')+20260805
                rng=np.random.default_rng(seed);draws=rng.choice(values,size=(2000,len(values))).mean(1)
                lo,hi=np.quantile(draws,[.025,.975]) if len(values)>=2 else [np.nan,np.nan]
                rows.append({'cohort':cohort,'scope':scope,'context':label,'n_sections':len(values),'n_events':len(g),**section.mean().to_dict(),'ci_low':lo,'ci_high':hi,'n_bootstrap':2000,'bootstrap_seed':seed,'independent_unit':'section','component_to_map_to_section_equal':True})
    pd.DataFrame(rows).to_csv(out/'CONTEXT_RESULTS.csv',index=False)

def external_lopo(run):
    events=pd.read_csv(run/'evaluation/external_events.csv.gz');rows=[]
    for d in sorted(events.dataset.unique()):
        values,_,_=summarize(events[events.dataset.ne(d)],seed=20260720);rows.append({'omitted_section':d,'bootstrap_seed':20260720,**values})
    out=run/'experiments/external_lopo';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'LOPO_RESULTS.csv',index=False)

def proposal_audit(run):
    from .legacy import greedy
    import csv,psutil
    f=core.functions();samples,_=manifests();out=run/'experiments/proposal';out.mkdir(parents=True,exist_ok=True);records=[];curves=[];counts=[];runtime=[];pathsdir=out/'growth_paths';pathsdir.mkdir(exist_ok=True)
    for sample in samples[samples.cohort.eq('development')].itertuples():
        ctx=context(sample);state=np.load(path(run,sample.dataset)/'preprocess/state.npz');scores=state['cfar'];graph=f.internal.adjacency(ctx['coords']);spots=ctx['spot_ids'];p0=pd.read_parquet(path(run,sample.dataset)/'candidates/pool.parquet')
        pools={p:dict((r.member_spots,(r.proposal_type,float(r.proposal_score),int(r.n_spots))) for r in p0[p0.n_spots.le(2)].itertuples()) for p in ['P0','P1','P2']}
        started=time.time()
        handle=gzip.open(pathsdir/(sample.dataset+'.csv.gz'),'wt',encoding='utf-8',newline='')
        fields=['dataset','seed_index','seed_spot','prefix_size','member_spots','added_spot_id','added_spot_anomaly','formal_path_proposal_score','frontier_size_before','frontier_size_after','step_compute_seconds','path_terminal_after_prefix','path_stop_reason']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();step_seconds=0.;cap_hits=0;prefixes=0;peak_rss=0
        for seed in range(len(spots)):
            growth=greedy.generate_full_path(seed,scores,graph,spots)
            cap_hits+=growth[-1]['path_stop_reason']=='engineering_ceiling'
            peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
            for step in growth:
                writer.writerow({'dataset':sample.dataset,'seed_index':seed,'seed_spot':spots[seed],**{k:step[k] for k in fields[3:]}})
                step_seconds+=step['step_compute_seconds'];prefixes+=1
                size=step['prefix_size']
                if size<3:continue
                for pool in ['P2']+(['P1'] if size<=10 else [])+(['P0'] if size in [3,4,5,8,10] else []):
                    pools[pool].setdefault(step['member_spots'],(f'core_{size}',step['formal_path_proposal_score'],size))
        handle.close();runtime.append({'dataset':sample.dataset,'seeds':len(spots),'accepted_prefixes':prefixes,'growth_step_cpu_wall_seconds_sum':step_seconds,'elapsed_seconds':time.time()-started,'peak_process_rss_bytes':peak_rss,'cap_hit_seeds':cap_hits,'cap_hit_rate':cap_hits/len(spots),'ceiling':64,'ceiling_is_biological_maximum':False})
        for pool,vals in pools.items():
            folder=out/'pools'/pool;folder.mkdir(parents=True,exist_ok=True)
            frame=pd.DataFrame([{'dataset_id':sample.dataset,'sample_id':sample.sample,'candidate_id':core.candidate_identity(sample.dataset,v[0],members.split(';')),'candidate_key':sample.dataset+'||'+v[0]+'||'+members,'candidate_row_key':core.candidate_identity(sample.dataset,v[0],members.split(';')),'proposal_type':v[0],'member_spots':members,'canonical_member_spots':members,'proposal_score':v[1],'n_spots':v[2]} for members,v in vals.items()])
            frame.to_parquet(folder/(sample.dataset+'.parquet'),index=False);counts.append({'dataset':sample.dataset,'pool':pool,'candidate_count':len(frame),'seconds':time.time()-started})
        assert set(pools['P0'])==set(p0.member_spots)
        # Truth is accessed after the new proposal pools have been written.
        component=core.components(truth(sample),ctx,sample.source)
        index=defaultdict(set)
        for member in pools['P2']:
            for s in member.split(';'):index[s].add(member)
        for row in component.itertuples():
            if row.component_size_spots>10:continue
            true=set(row.truth_spots);possible=set().union(*(index[s] for s in true));ious={m:len(set(m.split(';'))&true)/len(set(m.split(';'))|true) for m in possible}
            for pool in ['P0','P1','P2']:
                best=max((ious[m] for m in possible if m in pools[pool]),default=0);exact=max((ious[m] for m in possible if m in pools[pool] and pools[pool][m][2]==len(true)),default=0)
                records.append({'dataset':sample.dataset,'component':row.truth_component_id,'truth_size':len(true),'pool':pool,'oracle_iou':best,'exact_size_oracle_iou':exact})
            for k in range(3,65):curves.append({'dataset':sample.dataset,'component':row.truth_component_id,'K':k,'oracle_iou':max((ious[m] for m in possible if pools['P2'][m][2]<=k),default=0)})
        print('PROPOSAL_AUDIT',sample.dataset,{k:len(v) for k,v in pools.items()},flush=True)
    pd.DataFrame(runtime).to_csv(out/'RUNTIME_AND_CAP_HITS.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'CANDIDATE_COUNTS.csv',index=False);pd.DataFrame(records).to_csv(out/'ORACLE_COMPONENTS.csv.gz',index=False,compression='gzip')
    curves=pd.DataFrame(curves);curves.to_csv(out/'SATURATION_COMPONENTS.csv.gz',index=False,compression='gzip')
    section=curves.groupby(['dataset','K']).oracle_iou.mean().unstack();mean=section.mean();full=section[64];kstar=64
    for k in range(3,65):
        if mean[64]-mean[k]<=.001 and (full-section[k]).le(.01).sum()>=11:kstar=k;break
    dump(out/'SATURATION_RESULT.json',{'K_star':int(kstar),'global_loss_tolerance':.001,'section_loss_tolerance':.01,'minimum_sections':11,'engineering_ceiling':64,'new_parameters_selected_for_formal_pipeline':False})
    mean.rename('sample_equal_oracle_iou').to_csv(out/'SATURATION_CURVE.csv')

def cfar_biology(run):
    sys.path.insert(0,str(PROJECT/'vendor'))
    from .legacy import bio
    f=core.functions();samples,pairs=manifests();out=run/'biology';out.mkdir(exist_ok=True)
    for d in ['samples','candidates','pathways']:(out/'intermediate'/d).mkdir(parents=True,exist_ok=True)
    bio.PROJECT=out;bio.GENESETS={name:next((PROJECT/'data/gene_sets').glob(pattern)) for name,pattern in [('hallmark','h.all.*'),('reactome','c2.cp.reactome.*'),('gobp','c5.go.bp.*')]}
    import statsmodels.formula.api as smf
    bio.smf=smf
    bio.SAMPLE_SPECS={s.dataset:(s.cohort,'PDAC' if s.dataset.startswith('ST_PDAC_') else 'kidney cancer' if s.dataset in ['A02','A03','A04'] else 'lung cancer' if s.dataset in ['A05','A06','A07','A08','A09'] else 'breast cancer',None) for s in samples.itertuples()}
    cmap=context_mapping();allmeta=[];fidelity=[]
    for sample in samples[samples.cohort.ne('control')].itertuples():
        ctx=context(sample);state=np.load(path(run,sample.dataset)/'preprocess/state.npz')
        spec={'dataset_id':sample.dataset,'expression':state['normalized_hvg'],'genes':state['genes'],'spot_ids':state['spots'],'coords':ctx['coords'],'source_kind':'56C' if sample.source=='internal' else '72K1'}
        result=bio.compute_frontend(spec,f.internal,f.ext);direct=np.zeros_like(result['standardized'],dtype=np.float32)
        for i,bg in enumerate(result['backgrounds']):
            if len(bg)>=4:direct[i]=result['standardized'][i]-np.median(result['standardized'][bg],axis=0)
        rho=bio.vectorized_row_spearman(result['gene_signed'],direct);result['fidelity_rho']=rho.astype(np.float32);bio.save_sample_npz(result)
        difference=float(np.max(np.abs(result['formal_function_score']-state['cfar'])))
        if difference>1e-7:raise RuntimeError('Fresh CFAR detector disagreement: '+sample.dataset)
        fidelity.append({'sample':sample.dataset,'n_spots':len(rho),'median_spearman':np.nanmedian(rho),'mean_spearman':np.nanmean(rho),'q25_spearman':np.nanquantile(rho,.25),'q75_spearman':np.nanquantile(rho,.75),'fraction_positive':np.nanmean(rho>0),'max_absolute_detector_difference':difference,'detector_agreement_tolerance':1e-7})
        frames=[]
        for row in pairs[pairs.dataset.eq(sample.dataset)].itertuples():
            part=pd.read_csv(path(run,sample.dataset,row.method)/'inference/selected_regions.csv');part['method']=row.method;frames.append(part)
        selected=pd.concat(frames,ignore_index=True);rows=[];ref=truth(sample);comps=core.components(ref,ctx,sample.source)
        lookup={s:(r.truth_label,r.truth_component_id) for r in comps.itertuples() for s in r.truth_spots};spot_to_idx={s:i for i,s in enumerate(ctx['spot_ids'])}
        cancer='PDAC' if sample.dataset.startswith('ST_PDAC_') else 'kidney cancer' if sample.dataset in ['A02','A03','A04'] else 'lung cancer' if sample.dataset in ['A05','A06','A07','A08','A09'] else 'breast cancer'
        stratum='development' if sample.cohort=='development' else 'held-out external'
        for members,g in selected.groupby('member_spots',sort=False):
            rr=g.iloc[0];ids=members.split(';');ix=[spot_to_idx[s] for s in ids];center=ctx['coords'][ix].mean(0);dist=np.linalg.norm(ctx['coords']-center,axis=1)
            bg=[int(i) for i in np.argsort(dist) if int(i) not in set(ix)][:30]
            votes=Counter((cmap[(stratum,lookup[s][0])],lookup[s][1]) for s in ids if s in lookup and cmap.get((stratum,lookup[s][0])) not in [None,'','unassigned'])
            assigned=sorted(votes.items(),key=lambda v:(-v[1],v[0][0],v[0][1]))[0][0][0] if votes else 'unassigned'
            rows.append({'candidate_id':rr.candidate_id,'candidate_key':rr.candidate_key,'proposal_type':rr.proposal_type,'member_spots':members,'background_spots':';'.join(ctx['spot_ids'][i] for i in bg),'n_background':len(bg),'candidate_size':len(ids),'cohort':stratum,'cancer_type':cancer,'biological_context':assigned,'method_recurrence':g.method.nunique(),'methods':';'.join(sorted(g.method.unique())),'candidate_molecular_score':g.selection_score.median()})
        group=pd.DataFrame(rows);pool=pd.read_parquet(path(run,sample.dataset)/'candidates/pool.parquet');meta,audit=bio.save_candidate_npz(sample.dataset,group,result,pool)
        assert audit['missing_source_candidates']==0;allmeta.append(meta);print('CFAR_DECOMPOSED',sample.dataset,len(meta),flush=True)
    meta=pd.concat(allmeta,ignore_index=True);meta.to_csv(out/'CANONICAL_REGIONS.csv.gz',index=False,compression='gzip');pd.DataFrame(fidelity).to_csv(out/'GENE_CONCORDANCE.csv',index=False)
    section,landscape=bio.section_gene_and_landscape(meta);pc=bio.pc_landscape(meta);bio.pc_gene_drivers(pc)
    libraries,master=bio.aggregate_pathways(section,landscape);selected=bio.select_followup_terms(master,libraries,set(landscape[landscape.eligible_for_pathway].gene))
    corr,payload,contexts,cancers=bio.candidate_pathway_stage(meta,selected,libraries);bio.summarize_pathway_associations(corr,payload,selected)
    bio.finalize_pathway_classes(master,contexts,cancers);bio.random_calibration(meta,selected,libraries);bio.candidate_explanations(meta);bio.select_representative(meta,pd.DataFrame(fidelity))
    dump(out/'BIOLOGY_STATUS.json',{'status':'FRESH','sections':meta['sample'].nunique(),'canonical_regions':len(meta),'historical_regions_used':False,'section_permutations':10000,'gsea_permutations':1000,'random_regions_per_section_size':500})

def dense_selector(run):
    from .dense_development import execute
    execute(run)
