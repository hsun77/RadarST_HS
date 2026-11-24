from __future__ import annotations
from pathlib import Path
import json,textwrap,re
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from .pipeline import PROJECT,dump

BLUE='#397fac';ORANGE='#cf8424';GREY='#c9cdd1'
SHORT={'TLS/lymphoid microdomain':'TLS / lymphoid','Malignant epithelial/tumor':'Tumor','Immune/inflammatory context':'Immune / inflammatory','Stromal/benign tissue context':'Stromal / benign'}


def examples(run,meta,payload):
    protocol=json.loads((PROJECT/'configs/figure6_case_protocol.json').read_text())
    images=pd.read_csv(PROJECT/'manifests/HISTOLOGY_IMAGES.csv');finite=payload.groupby('candidate_key').cfar_pathway_attribution.count()
    explanations={}
    with (run/'biology/candidate_cfar_explanations.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line);explanations[row['candidate_key']]=row
    eligible=meta[meta.candidate_size.between(3,8)&meta.formal_proposal_score.notna()&meta.n_background.ge(4)&meta.member_spots.notna()&meta.background_spots.notna()&meta['sample'].isin(images.dataset)&meta.candidate_key.isin(explanations)&meta.candidate_key.map(finite).fillna(0).gt(0)&meta.median_member_spot_attribution_fidelity.ge(meta.section_median_spot_attribution_fidelity)].copy()
    initial=eligible[eligible.candidate_size.le(6)&eligible.workflow_recurrence_count.ge(2)&eligible.candidate_key.map(finite).ge(2)]
    anchors={}
    for context in protocol['initial_contexts']:
        pool=initial[initial.biological_context.eq(context)].sort_values(['sample','candidate_key'],kind='mergesort')
        if pool.empty:raise RuntimeError('Original context-example eligibility has no fresh candidate: '+context)
        anchors[context]=pool.iloc[0]
    chosen=[];used_keys=set();used_samples=set();used_tissues=set()
    for context,quota in protocol['atlas_quotas'].items():
        if context in anchors:
            row=anchors[context];chosen.append(row);used_keys.add(row.candidate_key);used_samples.add(row['sample']);used_tissues.add(row.cancer_type);quota-=1
        for _ in range(quota):
            pool=eligible[eligible.biological_context.eq(context)&~eligible.candidate_key.isin(used_keys)].copy()
            pool['new_tissue']=~pool.cancer_type.isin(used_tissues);pool['new_sample']=~pool['sample'].isin(used_samples)
            pool=pool.sort_values(['new_tissue','new_sample','sample','candidate_key'],ascending=[False,False,True,True],kind='mergesort')
            if pool.empty:raise RuntimeError('Fixed atlas quota unavailable: '+context)
            row=pool.iloc[0];chosen.append(row);used_keys.add(row.candidate_key);used_samples.add(row['sample']);used_tissues.add(row.cancer_type)
    selected=pd.DataFrame(chosen).reset_index(drop=True);selected.to_csv(run/'figures/FIGURE6_SELECTED_CASES.csv',index=False)
    dump(run/'figures/FIGURE6_CASE_LOCK.json',{'rule':protocol,'selected':selected[['sample','candidate_key','candidate_size','biological_context']].to_dict('records'),'gene_or_pathway_magnitude_used_to_choose_examples':False})
    rows=[]
    for i,row in selected.iterrows():
        axes=payload[payload.candidate_key.eq(row.candidate_key)&payload.cfar_pathway_attribution.notna()].copy();axes['absolute']=axes.cfar_pathway_attribution.abs();axes=axes.sort_values(['absolute','library','term'],ascending=[False,True,True],kind='mergesort')
        archive=np.load(run/'biology/intermediate/candidates'/f"{row['sample']}.npz")
        indices=np.flatnonzero(archive['candidate_key'].astype(str)==row.candidate_key)
        if len(indices)!=1:raise RuntimeError('Regional gene archive identity is not unique')
        genes=pd.DataFrame({'gene':archive['genes'].astype(str),'attribution':archive['signed'][indices[0]]});genes['absolute']=genes.attribution.abs();genes=genes.sort_values(['absolute','gene'],ascending=[False,True],kind='mergesort')
        for rank,axis in enumerate(axes.itertuples(),1):rows.append({'example':i,'sample':row['sample'],'candidate_key':row.candidate_key,'context':row.biological_context,'family':'pathway','name':axis.library+':'+axis.term,'value':axis.cfar_pathway_attribution,'rank':rank,'main_displayed':rank<=3,'atlas_displayed':rank<=10})
        for rank,gene in enumerate(genes.itertuples(),1):rows.append({'example':i,'sample':row['sample'],'candidate_key':row.candidate_key,'context':row.biological_context,'family':'gene','name':gene.gene,'value':gene.attribution,'rank':rank,'main_displayed':rank<=3,'atlas_displayed':rank<=10})
    source=pd.DataFrame(rows);source.to_csv(run/'figures/FIGURE6_REGION_DECOMPOSITIONS.csv',index=False)
    return selected,source


def mini(ax,rows,limit,title=None):
    rows=rows.sort_values('rank');names=[textwrap.fill(n.replace('HALLMARK_','').replace('REACTOME_','').replace('GOBP_','').replace('_',' '),38) for n in rows.name]
    ax.barh(np.arange(len(rows)),rows.value,color=[BLUE if v<0 else ORANGE for v in rows.value],height=.55)
    ax.axvline(0,c='#555',lw=.5);ax.set_xlim(-limit,limit);ax.set_yticks(np.arange(len(rows)),names,fontsize=5.8);ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_xticks([])
    if title:ax.set_title(title,fontsize=8)


def crop(ax,run,row,title=True):
    from .plots import histology
    z=np.load(run/'biology/intermediate/samples'/f"{row['sample']}.npz");ids=z['spot_ids'].astype(str);xy=z['coords'];core=np.isin(ids,row.member_spots.split(';'));bg=np.isin(ids,row.background_spots.split(';'))
    center=xy[core].mean(0);spacing=np.median(__import__('scipy').spatial.cKDTree(xy).query(xy,k=2)[0][:,1]);radius=4*spacing
    visible=(np.abs(xy[:,0]-center[0])<=radius)&(np.abs(xy[:,1]-center[1])<=radius)
    histology(ax,row['sample']);ax.scatter(xy[visible,0],xy[visible,1],c=GREY,s=6,linewidths=0);ax.scatter(xy[bg&visible,0],xy[bg&visible,1],c=BLUE,s=9,linewidths=0);ax.scatter(xy[core,0],xy[core,1],c=ORANGE,s=16,linewidths=.3,edgecolors='white')
    ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]+radius,center[1]-radius);ax.set_aspect('equal');ax.set_axis_off()
    if title:ax.set_title(SHORT[row.biological_context]+f"\n{row['sample']} | n={int(row.candidate_size)}",fontsize=6.7)


def term_name(name):
    value=name.split(':',1)[-1];library=name.split(':',1)[0] if ':' in name else ''
    for prefix in ['HALLMARK_','REACTOME_','GOBP_']:
        if value.startswith(prefix):
            if not library:library={'HALLMARK_':'hallmark','REACTOME_':'reactome','GOBP_':'gobp'}[prefix]
            value=value[len(prefix):]
    label=value.replace('_',' ').capitalize();label=re.sub(r'\b(atp|dna|rna|ecm|igf|igfbps|tp53)\b',lambda m:m.group().upper(),label)
    return '['+{'hallmark':'H','reactome':'R','gobp':'G'}[library]+'] '+label


def display_bars(fig,rect,rows,limit,label_rect,gene=False):
    rows=rows.sort_values('rank');ax=fig.add_axes(rect);labels=fig.add_axes(label_rect)
    ax.barh(range(len(rows)),rows.value,color=[BLUE if v<0 else ORANGE for v in rows.value],height=.52);ax.axvline(0,c='#555',lw=.5);ax.set_xlim(-limit,limit);ax.set_ylim(len(rows)-.5,-.5);ax.set_axis_off()
    labels.set(xlim=(0,1),ylim=(len(rows)-.5,-.5));labels.set_axis_off()
    for i,name in enumerate(rows.name):labels.text(1,i,name if gene else textwrap.fill(term_name(name),38),ha='right',va='center',fontsize=6.3 if gene else 5.5,linespacing=1.02)


def make_section_diagnostic(run):
    meta=pd.read_csv(run/'biology/CANONICAL_REGIONS.csv.gz');payload=pd.read_parquet(run/'biology/intermediate/candidate_pathway_attribution.parquet');classes=pd.read_csv(run/'biology/intermediate/cfar_pathway_classification_master.csv');rand=pd.read_csv(run/'biology/cfar_random_calibration.csv')
    selected,source=examples(run,meta,payload);fig=plt.figure(figsize=(8.27,11.69))
    fig.text(.025,.975,'a  Molecular decomposition',fontsize=9,weight='bold');fig.text(.025,.947,'Guarded local PC deviation\n↓ detector-selected signed PCs\n↓ PCA gene loading back-projection\nGene and pathway attribution',va='top',fontsize=8.5,linespacing=1.4)
    ax=fig.add_axes([.68,.875,.29,.095]);cf=rand[rand.metric.eq('CFAR intensity')];ax.scatter(cf.random_median,cf.observed_median,s=11,c=BLUE);lim=max(cf.random_median.max(),cf.observed_median.max());ax.plot([0,lim],[0,lim],ls='--',c=GREY);ax.set(xlabel='Matched-random CFAR',ylabel='Retained-region CFAR',title='b  Section × exact size')
    ax=fig.add_axes([.435,.49,.505,.34]);labels=fig.add_axes([.025,.49,.40,.34]);labels.set_axis_off();supported=classes[~classes.pathway_class.eq('E NO SUPPORT')].sort_values(['pathway_class','library','term']);keys=pd.MultiIndex.from_frame(supported[['library','term']]);matrix=payload.groupby(['library','term','sample']).cfar_pathway_attribution.median().unstack('sample').reindex(keys)
    if matrix.empty:ax.text(.5,.5,'No supported pathway class under source criteria',ha='center',transform=ax.transAxes);ax.set_axis_off()
    else:
        names=[textwrap.fill(term_name(t),58) for _,t in matrix.index];edges=np.r_[0,np.cumsum([n.count('\n')+1.35 for n in names])];centers=(edges[1:]+edges[:-1])/2
        lim=float(np.nanmax(np.abs(matrix.to_numpy())));cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dedede');image=ax.pcolormesh(np.arange(len(matrix.columns)+1),edges,np.ma.masked_invalid(matrix.to_numpy()),cmap=cmap,vmin=-lim,vmax=lim,rasterized=False);ax.set_ylim(edges[-1],0);ax.set_yticks([])
        short=lambda s:s.split('_')[-1] if s.startswith('EXT04_') else s.replace('ST_PDAC_','PDAC ').replace('HER2ST_','HER2 ')
        ax.set_xticks(np.arange(len(matrix.columns))+.5,[short(s) for s in matrix.columns],rotation=90,fontsize=5.5);ax.tick_params(length=2);labels.set(xlim=(0,1),ylim=(edges[-1],0))
        for y,name in zip(centers,names):labels.text(1,y,name,ha='right',va='center',fontsize=5.5,linespacing=1.05)
        fig.colorbar(image,cax=fig.add_axes([.95,.535,.012,.25]),label='Median signed attribution')
        pd.DataFrame({'library':matrix.index.get_level_values(0),'term':matrix.index.get_level_values(1),'display_label':names,'layout_band_height':np.diff(edges)}).to_csv(run/'figures/FIGURE6_LABEL_MAPPING.csv',index=False)
    fig.text(.025,.85,'c  Molecular axes across sections',fontsize=9,weight='bold');matrix.to_csv(run/'figures/FIGURE6_SECTION_PROGRAM_SOURCE.csv')
    fig.text(.025,.425,'d  Deterministically selected regional decompositions',fontsize=9,weight='bold');fig.text(.40,.405,'CFAR axes',fontsize=7,ha='center');fig.text(.82,.405,'Leading genes',fontsize=7,ha='center')
    pathway_limit=max(float(source.loc[source.family.eq('pathway')&source.main_displayed,'value'].abs().max()),1e-8);gene_limit=max(float(source.loc[source.family.eq('gene')&source.main_displayed,'value'].abs().max()),1e-8)
    for i,row in selected.iterrows():
        y=.345-i*.055;crop(fig.add_axes([.105,y,.115,.048]),run,row,title=False);fig.text(.025,y+.041,SHORT[row.biological_context]+f"\n{row['sample']}\nn={int(row.candidate_size)}",fontsize=5.4,va='top')
        for family,bar_rect,label_rect in [('pathway',[.54,y,.15,.048],[.24,y,.29,.048]),('gene',[.86,y,.105,.048],[.72,y,.13,.048])]:display_bars(fig,bar_rect,source[source.example.eq(i)&source.family.eq(family)&source.main_displayed],pathway_limit if family=='pathway' else gene_limit,label_rect,gene=family=='gene')
    fig.text(.025,.035,'Orange: retained core; blue: 30-spot region-level direct-contrast reference.\nBars: negative ← 0 → positive; shared scale within each feature family.\nH: Hallmark; R: Reactome; G: GO BP. Gray heatmap cells: attribution not estimable.',va='top',fontsize=6.1,linespacing=1.2)
    atlas,axes=plt.subplots(6,2,figsize=(12,18),layout='constrained')
    for i in range(6):
        for j,family in enumerate(['pathway','gene']):
            rows=source[source.example.eq(i)&source.family.eq(family)&source.atlas_displayed];limit=max(float(source.loc[source.family.eq(family)&source.atlas_displayed,'value'].abs().max()),1e-8);mini(axes[i,j],rows,limit,str(selected.iloc[i]['sample'])+' | '+family)
    for ext in ['svg','pdf','png']:atlas.savefig(run/'figures'/('Figure6_full_decomposition_atlas.'+ext),dpi=300,bbox_inches='tight')
    plt.close(atlas)
    return fig,['biology/CANONICAL_REGIONS.csv.gz','biology/intermediate/candidate_pathway_attribution.parquet','biology/intermediate/cfar_pathway_classification_master.csv','biology/cfar_random_calibration.csv','figures/FIGURE6_SELECTED_CASES.csv','figures/FIGURE6_REGION_DECOMPOSITIONS.csv','figures/FIGURE6_SECTION_PROGRAM_SOURCE.csv','figures/FIGURE6_LABEL_MAPPING.csv','manifests/HISTOLOGY_IMAGES.csv','configs/histology_display_policy.json'],'Current-region rules are re-applied, not historical candidate identities. Top3 axes/genes shown; all ranks and full top10 atlas exported. Blue region-level 30-spot reference is distinct from spot-level detector backgrounds. Gray cells are missing region-level attribution, never imputed zero. PDAC crops show actual array/grid coordinates without an unregistered H&E overlay. Variable band heights accommodate complete labels, not effect magnitudes. H/R/G distinguish gene-set libraries. No universal pathway activation or causality is inferred.'


def context_sources(run):
    classes=pd.read_csv(run/'biology/intermediate/cfar_pathway_classification_master.csv');context=pd.read_csv(run/'biology/cfar_pathway_by_context.csv');payload=pd.read_parquet(run/'biology/intermediate/candidate_pathway_attribution.parquet')
    supported=classes[classes.pathway_class.str.startswith(('B ','C '))][['library','term','pathway_class']]
    order=['TLS/lymphoid microdomain','Malignant epithelial/tumor','Immune/inflammatory context','Stromal/benign tissue context','Tumor boundary']
    assigned=context[context.biological_context.isin(order)].merge(supported,on=['library','term'],validate='many_to_one')
    ranking=assigned.groupby(['library','term','pathway_class'],as_index=False).agg(max_abs_context=('median_cfar_pathway_attribution',lambda x:x.abs().max()),context_span=('median_cfar_pathway_attribution',lambda x:x.max()-x.min())).sort_values(['max_abs_context','context_span','library','term'],ascending=[False,False,True,True],kind='mergesort').head(10)
    ranking['selection_rank']=np.arange(1,len(ranking)+1);chosen=assigned.merge(ranking[['library','term','selection_rank']],on=['library','term'],validate='many_to_one');chosen.to_csv(run/'figures/FIGURE6_CONTEXT_AXES.csv',index=False)
    programs=classes[classes.pathway_class.eq('C CONTEXT-ASSOCIATED CFAR PROGRAM')][['library','term','class_support_detail']].copy();programs['support_context']=programs.class_support_detail.str.split(';').str[0].str.strip()
    region=payload.merge(programs,on=['library','term'],validate='many_to_one');region=region[region.biological_context.eq(region.support_context)]
    recurrence=region.groupby(['support_context','library','term','sample'],as_index=False).agg(median_cfar_pathway_attribution=('cfar_pathway_attribution','median'),n_regions=('candidate_key','nunique'))
    center=context.merge(programs,on=['library','term'],validate='many_to_one');center=center[center.biological_context.eq(center.support_context)][['support_context','library','term','median_cfar_pathway_attribution']].rename(columns={'median_cfar_pathway_attribution':'locked_context_median'})
    recurrence=recurrence.merge(center,on=['support_context','library','term'],validate='many_to_one');keys=['support_context','library','term'];recurrence['same_direction']=np.sign(recurrence.median_cfar_pathway_attribution).eq(np.sign(recurrence.locked_context_median));recurrence['same_direction_n']=recurrence.groupby(keys).same_direction.transform('sum');recurrence['n_sections']=recurrence.groupby(keys)['sample'].transform('nunique');recurrence['n_finite_sections']=recurrence.groupby(keys).median_cfar_pathway_attribution.transform('count')
    ranks=center.copy();ranks['absolute']=ranks.locked_context_median.abs();ranks=ranks.sort_values(['support_context','absolute','library','term'],ascending=[True,False,True,True],kind='mergesort');ranks['display_rank']=ranks.groupby('support_context').cumcount()+1
    recurrence=recurrence.merge(ranks[keys+['display_rank']],on=keys,validate='many_to_one');recurrence['displayed']=recurrence.display_rank.le(6);recurrence.to_csv(run/'figures/FIGURE6_CONTEXT_RECURRENCE.csv',index=False)
    counts={name:int(classes.pathway_class.str.startswith(code+' ').sum()) for code,name in [('A','Broad'),('B','Heterogeneous'),('C','Context-associated'),('D','Cancer-associated')]};dump(run/'figures/FIGURE6_PROGRAM_COUNTS.json',counts)
    return ranking,chosen,recurrence,counts,order


def make(run):
    diagnostic,sources,note=make_section_diagnostic(run)
    for ext in ['pdf','svg','png']:diagnostic.savefig(run/'figures'/('Figure6_full_section_landscape.'+ext),dpi=250,bbox_inches='tight')
    plt.close(diagnostic);ranking,context,recurrence,counts,order=context_sources(run);selected=pd.read_csv(run/'figures/FIGURE6_SELECTED_CASES.csv');source=pd.read_csv(run/'figures/FIGURE6_REGION_DECOMPOSITIONS.csv');rand=pd.read_csv(run/'biology/cfar_random_calibration.csv')
    fig=plt.figure(figsize=(8.27,11.69));fig.text(.025,.98,'a  Molecular decomposition',fontsize=9,weight='bold');fig.text(.025,.95,'Guarded local PC deviation\n↓ detector-selected signed PCs\n↓ PCA gene loading back-projection\nGene and pathway attribution',va='top',fontsize=8.5,linespacing=1.4)
    ax=fig.add_axes([.68,.875,.29,.095]);cf=rand[rand.metric.eq('CFAR intensity')];ax.scatter(cf.random_median,cf.observed_median,s=11,c=BLUE);limit=max(cf.random_median.max(),cf.observed_median.max());ax.plot([0,limit],[0,limit],ls='--',c=GREY);ax.set(xlabel='Matched-random CFAR',ylabel='Retained-region CFAR',title='b  Section × exact size')
    fig.text(.025,.84,'c  Context-dependent molecular axes',fontsize=9,weight='bold');keys=pd.MultiIndex.from_frame(ranking[['library','term']]);matrix=context.pivot(index=['library','term'],columns='biological_context',values='median_cfar_pathway_attribution').reindex(keys).reindex(columns=order)
    labels=fig.add_axes([.025,.625,.39,.17]);labels.set_axis_off();labels.set(xlim=(0,1),ylim=(len(matrix)-.5,-.5))
    for i,(lib,term) in enumerate(matrix.index):labels.text(1,i,textwrap.fill(term_name(lib+':'+term),75),ha='right',va='center',fontsize=5.5,linespacing=1.02)
    ax=fig.add_axes([.425,.625,.35,.17]);cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dedede');limit=max(.20,float(np.nanmax(np.abs(matrix.to_numpy()))));mesh=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),aspect='auto',cmap=cmap,vmin=-limit,vmax=limit);ax.set_yticks([]);ax.set_xticks(range(5),['TLS /\nlymphoid','Tumor','Immune /\ninflammatory','Stromal /\nbenign','Tumor\nboundary'],fontsize=5.5);ax.xaxis.tick_top();ax.tick_params(length=0);fig.colorbar(mesh,cax=fig.add_axes([.425,.598,.35,.01]),orientation='horizontal',label='Signed CFAR attribution')
    for i,(name,value) in enumerate(counts.items()):fig.text(.81,.78-i*.03,f'{name}: {value}',fontsize=6.5)
    fig.text(.025,.565,'d  Deterministically selected regional decompositions',fontsize=9,weight='bold');fig.text(.40,.545,'CFAR axes',fontsize=7,ha='center');fig.text(.82,.545,'Leading genes',fontsize=7,ha='center')
    limits={family:max(float(source.loc[source.family.eq(family)&source.main_displayed,'value'].abs().max()),1e-8) for family in ['pathway','gene']}
    for i,row in selected.iterrows():
        y=.48-i*.055;crop(fig.add_axes([.105,y,.115,.048]),run,row,title=False);fig.text(.025,y+.041,textwrap.fill(SHORT[row.biological_context],16)+f"\n{row['sample']}\nn={int(row.candidate_size)}",fontsize=5.4,va='top')
        for family,bar_rect,label_rect in [('pathway',[.54,y,.15,.048],[.24,y,.29,.048]),('gene',[.86,y,.105,.048],[.72,y,.13,.048])]:display_bars(fig,bar_rect,source[source.example.eq(i)&source.family.eq(family)&source.main_displayed],limits[family],label_rect,gene=family=='gene')
    fig.text(.025,.187,'e  Context-associated program recurrence across sections',fontsize=9,weight='bold');visible=recurrence[recurrence.displayed];contexts=list(visible.support_context.drop_duplicates())
    for n,ctx in enumerate(contexts):
        left=.02+n*(.96/len(contexts));width=.96/len(contexts);sub=visible[visible.support_context.eq(ctx)];program_order=sub[['library','term','display_rank','same_direction_n','n_sections']].drop_duplicates().sort_values(['display_rank','library','term']);index=pd.MultiIndex.from_frame(program_order[['library','term']]);m=sub.pivot(index=['library','term'],columns='sample',values='median_cfar_pathway_attribution').reindex(index)
        label=fig.add_axes([left,.065,width*.48,.09]);label.set_axis_off();label.set(xlim=(0,1),ylim=(len(m)-.5,-.5));ax=fig.add_axes([left+width*.50,.065,width*.37,.09]);lim=max(.18,float(visible.median_cfar_pathway_attribution.abs().max()));ax.imshow(np.ma.masked_invalid(m.to_numpy()),aspect='auto',cmap=cmap,vmin=-lim,vmax=lim);ax.set_yticks([]);ax.set_xticks(range(len(m.columns)),m.columns,rotation=90,fontsize=5);ax.tick_params(length=0)
        for i,((lib,term),r) in enumerate(zip(m.index,program_order.itertuples())):label.text(1,i,textwrap.fill(term_name(lib+':'+term),42),ha='right',va='center',fontsize=5,linespacing=1.02);ax.text(len(m.columns),i,f'{int(r.same_direction_n)}/{int(r.n_sections)}',fontsize=5,va='center')
        ax.set_title(SHORT[ctx]+' | '+str(m.shape[1])+' sections',fontsize=6.4)
    fig.text(.025,.022,'Orange: retained core; blue: region-level 30-spot reference. Signed bars share a scale within each feature family.\nH: Hallmark; R: Reactome; G: GO BP. Gray cells are not estimable, not zero. Recurrence fractions are descriptive.\nPDAC: actual grid coordinates, without an unregistered H&E overlay.',fontsize=5.8,va='top')
    return fig,sources+['biology/cfar_pathway_by_context.csv','figures/FIGURE6_CONTEXT_AXES.csv','figures/FIGURE6_CONTEXT_RECURRENCE.csv','figures/FIGURE6_PROGRAM_COUNTS.json'],note+' Main c follows source top10 max absolute context-median/context-span ordering; e follows top6 per support context and reports actual section counts, without historical 9-program/8-section/9-section assertions.'
