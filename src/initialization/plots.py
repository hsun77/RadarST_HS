from __future__ import annotations
from pathlib import Path
import json,string,textwrap,traceback
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
from .pipeline import PROJECT,manifests,path,context,truth,base,sha,dump
from . import core

BLUE='#397fac';ORANGE='#cf8424';GREY='#a6acb3'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.facecolor':'white'})


def spatial(ax,ctx,labels,target=None,limits=None,color_map=None):
    codes=pd.factorize(np.asarray(labels).astype(str),sort=True)[0]
    colors=[color_map.get(str(v),ORANGE) for v in labels] if color_map else codes
    ax.scatter(ctx['coords'][:,0],ctx['coords'][:,1],c=colors,cmap=None if color_map else 'tab20',s=4,linewidths=0)
    if target:
        mask=np.isin(ctx['spot_ids'],list(target));ax.scatter(ctx['coords'][mask,0],ctx['coords'][mask,1],facecolors='none',edgecolors=ORANGE,s=30,linewidths=1)
    ax.set_aspect('equal')
    if not ax.yaxis_inverted():ax.invert_yaxis()
    ax.set_axis_off()
    if limits:ax.set_xlim(limits[:2]);ax.set_ylim(limits[3],limits[2])


def histology(ax,dataset):
    policy=PROJECT/'configs/histology_display_policy.json'
    if policy.exists() and dataset in json.loads(policy.read_text())['disabled']:return False
    manifest=PROJECT/'manifests/HISTOLOGY_IMAGES.csv'
    if not manifest.exists():return False
    table=pd.read_csv(manifest);selected=table[table.dataset.eq(dataset)]
    if len(selected)!=1:return False
    record=selected.iloc[0];image=plt.imread(PROJECT/record.image);sf=float(record.scale_factor)
    ax.imshow(image,extent=(0,image.shape[1]/sf,image.shape[0]/sf,0),zorder=-2)
    return True


def heat(ax,frame,label,cmap='Blues',center=False):
    masked=np.ma.masked_invalid(frame.to_numpy(float));v=np.nanmax(np.abs(masked)) if center else None
    im=ax.imshow(masked,aspect='auto',cmap=cmap,vmin=-v if center else None,vmax=v if center else None)
    ax.set_xticks(range(len(frame.columns)),list(map(str,frame.columns)),rotation=90);ax.set_yticks(range(len(frame.index)),list(map(str,frame.index)))
    ax.figure.colorbar(im,ax=ax,shrink=.65,pad=.02,label=label);return im


def small_events(run,cohort):
    ev=pd.read_csv(run/'evaluation'/(cohort+'_events.csv.gz'));return ev[ev.truth_size.between(1,10)].copy()


def figure1(run):
    samples,pairs=manifests();fig,axes=plt.subplots(3,1,figsize=(7.2,6.3),gridspec_kw={'height_ratios':[1.5,2,1]});fig.suptitle('RadarST: training and single-upstream refinement',fontsize=12)
    axes[0].set_axis_off();texts=['Raw expression\nSpatial coordinates','One upstream\ncluster map','Guarded CFAR\nConnected\nproposals','Single-map features\nCandidate ranking','Selector\nFinal labels']
    for i,label in enumerate(texts):
        x=.015+i*.20;axes[0].add_patch(Rectangle((x,.48),.17,.25,transform=axes[0].transAxes,facecolor='#eaf2f7',edgecolor=BLUE,lw=.8));axes[0].text(x+.085,.605,label,transform=axes[0].transAxes,ha='center',va='center',fontsize=6.8)
        if i in [2,3]:axes[0].annotate('',xy=(x+.20,.605),xytext=(x+.17,.605),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':BLUE,'lw':.9})
    axes[0].plot([.10,.10,.50,.50],[.73,.82,.82,.75],color=BLUE,lw=.9,transform=axes[0].transAxes);axes[0].annotate('',xy=(.50,.73),xytext=(.50,.79),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':BLUE,'lw':.9})
    axes[0].plot([.30,.30,.70,.70],[.48,.37,.37,.46],color=GREY,lw=.9,transform=axes[0].transAxes);axes[0].annotate('',xy=(.70,.48),xytext=(.70,.40),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':GREY,'lw':.9})
    axes[0].text(.5,.94,'12 LOSO fits + Development-only full fit; External used for evaluation',ha='center',transform=axes[0].transAxes,fontsize=7)
    axes[0].text(.5,.15,'Reference annotations: training targets and post-inference evaluation only\nCompleted upstream maps are supplied, not rerun.',ha='center',transform=axes[0].transAxes,fontsize=7)
    cohorts=samples.cohort.value_counts().reindex(['control','development','external']);axes[1].bar(cohorts.index,cohorts.values,color=[GREY,BLUE,ORANGE]);axes[1].set_ylabel('Sections');axes[1].set_title('Study cohorts and completed upstream maps')
    for i,v in enumerate(cohorts):axes[1].text(i,v+.2,str(v),ha='center')
    axes[2].set_axis_off();axes[2].text(.05,.7,'Component best-IoU = max |Truth ∩ Region| / |Truth ∪ Region|\nCoverage = TP / (TP + FN)       Purity = TP / (TP + FP)\nPrimary inference: 12×14 Development + 11×14 External; 12×14 control inputs',va='top',linespacing=1.8)
    return fig,['manifests/SAMPLES.csv','manifests/PAIRS.csv','configs/formal.json'], 'Schematic, not a computed patient image. Purity uses TP/(TP+FP).'


def figure2(run):
    samples,pairs=manifests();fig=plt.figure(figsize=(8.27,11.69),layout='constrained');grid=fig.add_gridspec(5,2,height_ratios=[1.35,1.9,1,1,.6]);axes=[]
    sub=grid[0,0].subgridspec(3,3);s=next(s for s in samples.itertuples() if s.dataset=='151673');ctx=context(s);rows=pairs[pairs.dataset.eq(s.dataset)].set_index('method')
    maps=[('Annotation',truth(s))]+[(m,base(rows.loc[m],ctx)) for m in ['GraphST','STAGATE','DeepST','SpaceFlow','SEDR','SpaGCN','BayesSpace','CCST']]
    for i,(label,values) in enumerate(maps):ax=fig.add_subplot(sub[i//3,i%3]);spatial(ax,ctx,values);ax.set_title(('a  ' if i==0 else '')+label,fontsize=7)
    ax=fig.add_subplot(grid[0,1]);macro=pd.read_csv(run/'evaluation/dlpfc_macro.csv');wide=macro.pivot(index='method',columns='dataset',values='ARI');wide['Median']=wide.median(axis=1);heat(ax,wide,'ARI');ax.axvline(11.5,color='white',lw=2);ax.set_title(f'b  Broad-domain agreement\nn=12 sections/workflow; pooled median={macro.ARI.median():.3f}',fontsize=8)
    hero=grid[1,:].subgridspec(1,4);s=next(s for s in samples.itertuples() if s.dataset=='10X_BC');ctx=context(s);r=next(r for r in pairs.itertuples() if r.dataset==s.dataset and r.method=='GraphST');up=base(r,ctx);ev=small_events(run,'development');cand=ev[ev.dataset.eq(s.dataset)&ev.method.eq('GraphST')&ev.truth_size.ge(2)].sort_values(['dilution_before','truth_component_id'],ascending=[False,True]).head(2)
    comps=core.components(truth(s),ctx,s.source).set_index('truth_component_id')
    cand.to_csv(run/'figures/FIGURE2_ZOOM_SOURCE.csv',index=False);full_axes=[]
    for i,label in enumerate(['c  Truth','GraphST','Zoom 1','Zoom 2']):
        ax=fig.add_subplot(hero[0,i])
        if i<2:
            histology(ax,s.dataset);spatial(ax,ctx,truth(s) if i==0 else up);full_axes.append(ax)
        else:
            event=cand.iloc[i-2];members=comps.loc[event.truth_component_id].truth_spots;mask=np.isin(ctx['spot_ids'],members);xy=ctx['coords'][mask];spacing=np.median(__import__('scipy').spatial.cKDTree(ctx['coords']).query(ctx['coords'],k=2)[0][:,1]);pad=spacing*3
            limits=[xy[:,0].min()-pad,xy[:,0].max()+pad,xy[:,1].min()-pad,xy[:,1].max()+pad];histology(ax,s.dataset);spatial(ax,ctx,up,members,limits)
            for full in full_axes:
                full.add_patch(Rectangle((limits[0],limits[2]),limits[1]-limits[0],limits[3]-limits[2],fill=False,edgecolor=ORANGE,lw=.9));full.text(limits[0],limits[2],str(i-1),color=ORANGE,fontsize=8,va='bottom')
            ax.set_xlabel(f'n={int(event.truth_size)}; coverage={event.coverage_before:.2f}\nIoU={event.before:.3f}',fontsize=7)
            ax.text(.5,-.05,f'n={int(event.truth_size)}; coverage={event.coverage_before:.2f}\nIoU={event.before:.3f}',ha='center',va='top',transform=ax.transAxes,fontsize=7)
        ax.set_title(label,fontsize=8)
    ax=fig.add_subplot(grid[2,0]);comp=pd.read_csv(run/'tables/FRESH_TRUTH_COMPONENTS.csv.gz');dev=comp[comp.cohort.eq('development')];sizes=dev.component_size_spots;ax.hist(sizes.clip(upper=30),bins=np.arange(.5,31.5),color=GREY);ax.axvspan(.5,10.5,color=BLUE,alpha=.12);ax.set(xlabel='Exact component size (30 = ≥30)',ylabel='Components',title='d  Exact-size distribution');ax.text(.97,.95,f'{len(sizes)} total; {sizes.le(10).sum()} size 1–10\n{sizes.le(10).mean():.1%}',transform=ax.transAxes,ha='right',va='top',fontsize=7)
    ax=fig.add_subplot(grid[2,1]);exact=ev.groupby(['method','dataset','truth_size']).before.mean().groupby(['method','truth_size']).mean().unstack().reindex(columns=range(1,11));heat(ax,exact,'Mean best-IoU');ax.set_title('e  Workflow × exact size')
    ax=fig.add_subplot(grid[3,0]);ax.scatter(ev.coverage_before,np.log2(ev.dilution_before.clip(lower=.01)),s=3,alpha=.12,color=GREY);ax.axvline(.8,color=BLUE,ls='--');ax.axhline(2,color=BLUE,ls='--');ax.set(xlabel='Coverage',ylabel='log₂(predicted/truth size)',title='f  Coverage versus dilution')
    ax=fig.add_subplot(grid[3,1]);pm=ev.groupby(['dataset','method']).before.mean().reset_index().merge(pd.read_csv(run/'evaluation/development_macro.csv'),on=['dataset','method'],validate='one_to_one');valid=pm[['ARI_before','before']].dropna();rho=stats.spearmanr(valid.ARI_before,valid.before).statistic;ax.scatter(valid.ARI_before,valid.before,s=9,c=GREY);summary=pm.groupby('method')[['ARI_before','before']].median();ax.scatter(summary.ARI_before,summary.before,marker='D',color=BLUE,s=22);ax.set(xlabel='Upstream ARI',ylabel='Compact best-IoU',title=f'g  Pair-level correspondence (n={len(valid)}; ρ={rho:.2f})')
    ev.assign(high_coverage_high_dilution=ev.coverage_before.ge(.8)&ev.dilution_before.ge(4)).to_csv(run/'figures/Figure2_threshold_source.csv.gz',index=False,compression='gzip')
    ax=fig.add_subplot(grid[4,:]);ranks=summary.rank(ascending=False);ranks.to_csv(run/'figures/Figure2_workflow_ranks.csv')
    for method,row in ranks.iterrows():
        ax.plot([0,1],[row.ARI_before,row.before],'-o',color=GREY,ms=3)
        ax.text(-.03,row.ARI_before,method,ha='right',fontsize=6);ax.text(1.03,row.before,method,ha='left',fontsize=6)
    ax.set(xlim=(-.5,1.5),ylim=(14.5,.5),xticks=[0,1],xticklabels=['Macro ARI rank','Compact IoU rank'],title='h  Workflow rank discordance');ax.set_yticks([])
    return fig,['evaluation/dlpfc_macro.csv','evaluation/development_events.csv.gz','evaluation/development_macro.csv','tables/FRESH_TRUTH_COMPONENTS.csv.gz','figures/Figure2_threshold_source.csv.gz','figures/Figure2_workflow_ranks.csv','figures/FIGURE2_ZOOM_SOURCE.csv','manifests/HISTOLOGY_IMAGES.csv','configs/histology_display_policy.json'],'Spatial maps preserve coordinate orientation. Colors identify source labels within each map and are not layer-aligned across methods. Numbered boxes locate the unchanged deterministic zooms; zoom outlines mark complete fresh truth members. The 0.8 coverage/4× dilution thresholds match actual data selection; workflow ranks are calculated from median pair-level endpoints. The median control column is a summary, not another section.'


def figure3(run):
    fig,axes=plt.subplots(3,2,figsize=(7.2,9),layout='constrained')
    for col,cohort in enumerate(['development','external']):
        ev=small_events(run,cohort);pair=ev.groupby(['dataset','method'])[['before','after','gain']].mean().reset_index();sec=pair.groupby('dataset').gain.mean();workflow=pair.groupby('method').gain.mean().sort_values()
        axes[0,col].barh(workflow.index,workflow,color=BLUE);axes[0,col].axvline(0,c='black',lw=.7);axes[0,col].set_title(cohort.capitalize());axes[0,col].set_xlabel('Section-equal gain')
        axes[1,col].scatter(pair.before,pair.after,s=12,c=BLUE,alpha=.6);axes[1,col].plot([0,1],[0,1],c=GREY,ls='--');axes[1,col].set(xlabel='Before',ylabel='After',xlim=(0,1),ylim=(0,1))
        exact=ev.groupby(['truth_size','dataset','method'])[['before','after']].mean().groupby(['truth_size','dataset']).mean().groupby('truth_size').mean().reindex(range(1,11))
        for metric,color in [('before',GREY),('after',BLUE)]:axes[2,col].plot(exact.index,exact[metric],'-o',color=color,label=metric,ms=3)
        axes[2,col].set(xlabel='Exact truth size',ylabel='Section-equal best-IoU',xticks=range(1,11));axes[2,col].legend()
        exact.to_csv(run/'figures'/(cohort+'_exact_size_source.csv'))
    return fig,['evaluation/development_events.csv.gz','evaluation/external_events.csv.gz'],'All pairs retained; absent exact sizes remain NaN and break connecting lines. Workflow means have equal section weight.'


def select_cases(run):
    dev=small_events(run,'development');ext=small_events(run,'external');from .experiments import context_mapping
    mapping=context_mapping();rows=[];used=set()
    slots=[('development','TLS/lymphoid microdomain')]*3+[('development','Immune/inflammatory context'),('development','Stromal/benign tissue context'),('development','Malignant epithelial/tumor')]+[('external','ANY')]*3
    for cohort,ctx in slots:
        g=(dev if cohort=='development' else ext).copy();stratum='development' if cohort=='development' else 'held-out external';g['context']=[mapping.get((stratum,t),'unassigned') for t in g.truth_label];g['dilution_reduction']=g.dilution_before-g.dilution_after
        g=g[g.method.eq('GraphST')&g.truth_size.between(2,10)&g.gain.gt(1e-12)&~g.dataset.isin(used)]
        if ctx!='ANY':g=g[g.context.eq(ctx)]
        else:g=g[g.context.eq('Malignant epithelial/tumor')]
        if g.empty:raise RuntimeError(f'Fixed Figure4 slot unavailable in fresh outcomes: {cohort} {ctx}; no replacement case selected')
        rr=g.sort_values(['gain','dilution_reduction','dataset','truth_component_id'],ascending=[False,False,True,True]).iloc[0];rows.append(rr);used.add(rr.dataset)
    return pd.DataFrame(rows)


def figure4(run):
    cases=select_cases(run);cases.to_csv(run/'figures/FIGURE4_SELECTED_CASES.csv',index=False);samples,pairs=manifests();fig=plt.figure(figsize=(8.27,11.69),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[4.7,1.3],wspace=.4);maps=grid[0,:].subgridspec(9,3,wspace=.02,hspace=.08);axes=np.array([[fig.add_subplot(maps[i,j]) for j in range(3)] for i in range(9)])
    for i,case in cases.iterrows():
        # Row order is fixed by the slot list, not the source event index.
        row_number=list(cases.index).index(i);s=next(s for s in samples.itertuples() if s.dataset==case.dataset);ctx=context(s);r=next(r for r in pairs.itertuples() if r.dataset==s.dataset and r.method=='GraphST');up=base(r,ctx);after=pd.read_csv(path(run,s.dataset,'GraphST')/'inference/final_labels.csv',dtype=str).set_index('spot_id').pred_label.reindex(ctx['spot_ids']);comp=core.components(truth(s),ctx,s.source).set_index('truth_component_id');members=comp.loc[case.truth_component_id].truth_spots
        xy=ctx['coords'][np.isin(ctx['spot_ids'],members)];span=max(np.ptp(xy[:,0]),np.ptp(xy[:,1]),float(np.median(__import__('scipy').spatial.cKDTree(ctx['coords']).query(ctx['coords'],k=2)[0][:,1]))*4);center=xy.mean(0);lim=[center[0]-2.2*span,center[0]+2.2*span,center[1]-span,center[1]+span]
        old_values=sorted(up.unique());color_map={v:plt.get_cmap('tab20')(i%20) for i,v in enumerate(old_values)}
        for j,labels in enumerate([truth(s),up,after]):histology(axes[row_number,j],s.dataset);spatial(axes[row_number,j],ctx,labels,members,lim,color_map if j else None)
        axes[row_number,0].text(-.04,.5,textwrap.fill(str(case.dataset),15)+f'\nn={int(case.truth_size)}',transform=axes[row_number,0].transAxes,ha='right',va='center',fontsize=6.5)
        axes[row_number,2].text(1.02,.5,f'{case.before:.3f} → {case.after:.3f}',transform=axes[row_number,2].transAxes,fontsize=7,va='center')
    for j,title in enumerate(['Reference annotations','GraphST upstream','RadarST']):axes[0,j].set_title(title)
    summary=pd.read_csv(run/'experiments/pathology/CONTEXT_RESULTS.csv');summary=summary[summary.scope.eq('size_1_10')&summary.context.ne('Tumor boundary')].copy();summary['display']=summary.cohort.str.capitalize()+': '+summary.context.map(lambda s:textwrap.fill(s,20));summary=summary.sort_values(['cohort','context']);ax=fig.add_subplot(grid[1,0]);y=np.arange(len(summary));ax.errorbar(summary.gain,y,xerr=np.vstack([summary.gain-summary.ci_low,summary.ci_high-summary.gain]),fmt='o',color=BLUE,ms=4,capsize=2);ax.set_yticks(y,summary.display,fontsize=6);ax.axvline(0,c=GREY,lw=.7);ax.set(xlabel='Equal-section gain (95% CI)',title='b  Compact-region context summaries');ax.invert_yaxis()
    sections=[]
    for cohort in ['development','external']:
        ev=pd.read_csv(run/'experiments/pathology'/f'{cohort}_context_events.csv.gz');ev=ev[ev.truth_size.between(1,10)&~ev.context.isin(['unassigned','Tumor boundary'])];part=ev.groupby(['context','dataset','method']).gain.mean().groupby(['context','dataset']).mean().reset_index();part['cohort']=cohort;sections.append(part)
    section=pd.concat(sections,ignore_index=True);section.to_csv(run/'figures/FIGURE4_SECTION_CONTEXT_SOURCE.csv',index=False);summary.to_csv(run/'figures/FIGURE4_CONTEXT_SOURCE.csv',index=False);ax=fig.add_subplot(grid[1,1])
    for n,row in enumerate(summary.itertuples()):
        values=section[section.cohort.eq(row.cohort)&section.context.eq(row.context)].gain;ax.scatter(values,np.full(len(values),n),s=13,color=BLUE if row.cohort=='development' else ORANGE,alpha=.7)
    ax.axvline(0,c=GREY,lw=.7);ax.set_yticks(y,summary.display,fontsize=6);ax.invert_yaxis();ax.set(xlabel='Section-context mean gain',title='c  Section-level context variation')
    return fig,['evaluation/development_events.csv.gz','evaluation/external_events.csv.gz','figures/FIGURE4_SELECTED_CASES.csv','manifests/HISTOLOGY_IMAGES.csv','configs/histology_display_policy.json','experiments/pathology/CONTEXT_RESULTS.csv','figures/FIGURE4_CONTEXT_SOURCE.csv','figures/FIGURE4_SECTION_CONTEXT_SOURCE.csv'],'Nine fixed context slots, GraphST, size2–10, positive gain; sort gain, dilution reduction, dataset/component; unique sections. This selection rule matches the source build protocol. Outcome-ranked illustrations are not unbiased validation. Original histology backgrounds are used only where image provenance and source scale factors are available and the grid-coordinate disable policy does not apply; missing or unregistered images are not simulated. Panels b/c use size1–10 equal-section estimates, 2,000 section-bootstrap draws, and descriptive section-context observations; multiple contexts from one section are not independent samples. The single-section boundary context is omitted from these inferential panels, not from source tables.'


def figure5(run):
    fig,axes=plt.subplots(3,2,figsize=(7.2,9),layout='constrained');primary=pd.read_csv(run/'tables/PRIMARY_RESULTS.csv')
    for i,cohort in enumerate(['development','external']):
        g=primary[primary.cohort.eq(cohort)];positions=np.arange(len(g))+i*3
        axes[0,0].bar(positions,g.gain,color=[BLUE,ORANGE]);axes[0,0].errorbar(positions,g.gain,yerr=np.vstack([g.gain-g.ci_low,g.ci_high-g.gain]),fmt='none',color='#333333',capsize=2,lw=.8);axes[0,0].set_xticks([.5,3.5],['Development','External']);axes[0,0].set_ylabel('Equal-section gain')
        for row in g.itertuples():axes[0,1].scatter(row.gain,row.loss_event_fraction,label=row.mode+' / '+cohort,s=45,marker='o' if cohort=='development' else 'D',color=BLUE if row.mode=='RadarST' else ORANGE)
        axes[0,1].set(xlabel='Mean gain',ylabel='Loss-event fraction');axes[0,1].legend(fontsize=6)
        p=run/'experiments/randomization'/(cohort+'_whole_benchmark_null.csv');null=pd.read_csv(p);axes[1,i].hist(null.sample_equal_gain,bins=35,color=GREY);obs=g[g['mode'].eq('RadarST')].gain.iloc[0];axes[1,i].axvline(obs,c=BLUE);axes[1,i].set_title(cohort+' randomization');axes[1,i].set_xlabel('Whole-benchmark sample-equal gain')
    grid=pd.read_csv(run/'experiments/sensitivity/GRID_RESULTS.csv');heat(axes[2,0],grid.pivot(index='expression_weight',columns='retention',values='gain'),'Gain');axes[2,0].set_title('Development sensitivity')
    macro=pd.concat([pd.read_csv(run/'evaluation'/f'{c}_macro.csv') for c in ['development','external']]);macro['modified_fraction']=macro.modified_spots/macro.n_spots
    display=np.isfinite(macro.modified_fraction)&np.isfinite(macro.ARI_delta)&~(macro.modified_fraction.eq(0)&macro.ARI_delta.eq(0));macro['displayed_in_scatter']=display
    macro.to_csv(run/'figures/FIGURE5_MACRO_DISPLAY_SOURCE.csv',index=False)
    axes[2,1].scatter(macro.loc[display,'modified_fraction'],macro.loc[display,'ARI_delta'],c=GREY,s=7);axes[2,1].axhline(0,c=BLUE,lw=.7);axes[2,1].set(xlabel='Modified-spot fraction',ylabel='ARI change',title=f'Macro compatibility (display n={int(display.sum())})')
    return fig,['tables/PRIMARY_RESULTS.csv','experiments/randomization/development_whole_benchmark_null.csv','experiments/randomization/external_whole_benchmark_null.csv','experiments/sensitivity/GRID_RESULTS.csv','evaluation/development_macro.csv','evaluation/external_macro.csv','figures/FIGURE5_MACRO_DISPLAY_SOURCE.csv'],'Mean-gain error bars are 10,000 section-bootstrap 95% intervals. Risk fractions are descriptive event totals. Formal comparator differences and confidence intervals use paired section bootstrap, not event-level tests. The macro scatter omits exact (0,0) zero-change pairs and undefined ARI only for display; all 322 pairs remain in tables and aggregate calculations, with the display mask preserved in source data.'


def figure6(run):
    from .interpretation_figure import make
    return make(run)

def figure6_diagnostic_statistics_reference(run):
    fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained');meta=pd.read_csv(run/'biology/CANONICAL_REGIONS.csv.gz');con=pd.read_csv(run/'biology/GENE_CONCORDANCE.csv');rnd=pd.read_csv(run/'biology/cfar_random_calibration.csv');classes=pd.read_csv(run/'biology/intermediate/cfar_pathway_classification_master.csv')
    axes[0,0].set_axis_off();axes[0,0].text(.05,.85,'Signed local PC deviations\n↓ detector-selected top eight PCs\n↓ PCA gene loadings\nSigned gene contributions\n↓ pathway interpretation',va='top',fontsize=11,linespacing=1.7);axes[0,0].set_title('a  Detector-centered decomposition')
    cf=rnd[rnd.metric.eq('CFAR intensity')];axes[0,1].scatter(cf.random_median,cf.observed_median,s=15,c=BLUE);lim=max(cf.random_median.max(),cf.observed_median.max());axes[0,1].plot([0,lim],[0,lim],c=GREY,ls='--');axes[0,1].set(xlabel='Matched-random CFAR',ylabel='Retained-region CFAR',title='b  Section × size calibration')
    counts=classes.pathway_class.value_counts();axes[1,0].barh([textwrap.fill(t,25) for t in counts.index],counts,color=BLUE);axes[1,0].set_title('c  Actual pathway-class counts')
    axes[1,1].scatter(con.n_spots,con.median_spearman,c=BLUE,s=18);axes[1,1].axhline(0,c=GREY,ls='--');axes[1,1].set(xlabel='Section spots',ylabel='Median gene concordance',title='d  Attribution / direct contrast')
    return fig,['biology/CANONICAL_REGIONS.csv.gz','biology/GENE_CONCORDANCE.csv','biology/cfar_random_calibration.csv','biology/intermediate/cfar_pathway_classification_master.csv'],'Actual canonical count retained. Decomposition is post-inference attribution, not pathway causality or universal activation. Regional decomposition panels and manuscript-equivalent layout need separate QA.'


def supplementary(run,n):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),layout='constrained')
    if n==1:
        frame=pd.read_csv(run/'experiments/pathology/CONTEXT_RESULTS.csv');frame=frame[frame.scope.eq('size_1_10')]
        for i,cohort in enumerate(['development','external']):
            g=frame[frame.cohort.eq(cohort)];axes[i].barh([textwrap.fill(t,22) for t in g.context],g.gain,color=BLUE);axes[i].set_title(cohort);axes[i].set_xlabel('Equal-section gain')
        sources=['experiments/pathology/CONTEXT_RESULTS.csv']
    elif n==2:
        sources=[]
        for i,c in enumerate(['development','external']):
            source=f'experiments/randomization/{c}_whole_benchmark_null.csv';g=pd.read_csv(run/source);axes[i].hist(g.sample_equal_gain,35,color=GREY);axes[i].set_title(c);axes[i].set_xlabel('Randomized equal-section gain');sources.append(source)
    elif n==3:
        source='experiments/sensitivity/GRID_RESULTS.csv';g=pd.read_csv(run/source);heat(axes[0],g.pivot(index='expression_weight',columns='retention',values='gain'),'Gain');heat(axes[1],g.pivot(index='expression_weight',columns='retention',values='loss_event_fraction'),'Event loss fraction');sources=[source]
    else:
        a='experiments/proposal/SATURATION_CURVE.csv';b='experiments/proposal/ORACLE_COMPONENTS.csv.gz';g=pd.read_csv(run/a);axes[0].plot(g.K,g.sample_equal_oracle_iou,color=BLUE);axes[0].set(xlabel='Maximum emitted size K',ylabel='Equal-section oracle IoU');oracle=pd.read_csv(run/b)
        for pool,part in oracle.groupby('pool'):
            v=part.groupby(['truth_size','dataset']).oracle_iou.mean().groupby('truth_size').mean().reindex(range(1,11));axes[1].plot(v.index,v.values,'-o',label=pool,ms=3)
        axes[1].set(xlabel='Exact truth size',ylabel='Oracle best-IoU');axes[1].legend();sources=[a,b]
    return fig,sources,'Independent sections define aggregate weighting. See source tables for missing strata and null/CI definitions.'


def build(run):
    folder=run/'figures';folder.mkdir(exist_ok=True);registry=[];failures=[]
    for name,fn in [(f'Figure{i}',globals()[f'figure{i}']) for i in range(1,7)]+[(f'FigureS{i}',lambda r,i=i:supplementary(r,i)) for i in range(1,5)]:
        try:
            fig,sources,note=fn(run);fig.canvas.draw()
            for ext in ['svg','pdf','png']:fig.savefig(folder/(name+'.'+ext),dpi=300,bbox_inches='tight')
            plt.close(fig)
            for source in sources:
                p=(PROJECT/source) if source.startswith(('manifests/','configs/')) else run/source
                complete=True
                registry.append({'figure':name,'status':'FRESH' if complete else 'BLOCKED','fresh_artifact_generated':True,'source':source,'sha256':sha(p),'script':'src/radarst/plots.py:'+fn.__name__,'note':note,'visual_QA':'NOT VERIFIED','manuscript_equivalent_layout':'NOT VERIFIED','remaining_content':'' if complete else 'Histology/case-decomposition manuscript panels and required original-panel traceability remain incomplete'})
            (folder/(name+'_legend.md')).write_text(note,encoding='utf-8');print('FIGURE_GENERATED',name,flush=True)
        except Exception:
            error=traceback.format_exc();failures.append({'figure':name,'error':error});registry.append({'figure':name,'status':'BLOCKED','source':'','note':error});print(error,flush=True)
    pd.DataFrame(registry).to_csv(run/'reports/FIGURE_SOURCE_MANIFEST.csv',index=False);dump(folder/'BUILD_ERRORS.json',failures)
    if failures:raise RuntimeError('Figure builds incomplete; see BUILD_ERRORS.json')
