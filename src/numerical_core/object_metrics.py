"""Class-agnostic, UNKNOWN-censored spot-instance diagnostics (custom PQ variant).
Reference-side IoU and union geometry preserve the frozen RadarST definition.
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from independent_evaluator import regions,quality,COMPACT_BAD,MACRO_BAD,information_metrics
class ObjectEvaluator:
    def __init__(self,truth,adj):
        self.truth=np.asarray(truth,str);self.adj=adj;self.n=len(truth);self.known=~np.isin(np.char.lower(self.truth),list(COMPACT_BAD));self.refs=[s for s in regions(self.truth,adj) if str(self.truth[min(s)]).lower() not in COMPACT_BAD];self.sizes=np.array([len(s) for s in self.refs]);self.compact=self.sizes<=10;self.refowner=np.full(self.n,-1,int)
        for j,s in enumerate(self.refs):self.refowner[list(s)]=j
    def evaluate(self,labels,details=False):
        labels=np.asarray(labels);ps=regions(labels,self.adj);sizes=np.array([len(s) for s in ps]);owner=np.empty(self.n,int);known=np.ones(len(ps),bool)
        for j,s in enumerate(ps):owner[list(s)]=j;known[j]=bool(self.known[list(s)].all())
        valid=self.refowner>=0;codes=self.refowner[valid]*len(ps)+owner[valid];code,count=np.unique(codes,return_counts=True);ri=code//len(ps);pi=code%len(ps);iou=count/(self.sizes[ri]+sizes[pi]-count);good=iou>.5+1e-12;matches=[(int(a),int(b),float(v)) for a,b,v in zip(ri[good],pi[good],iou[good])]
        assert len({a for a,b,v in matches})==len(matches) and len({b for a,b,v in matches})==len(matches)
        matched_ref={a for a,b,v in matches};matched_pred={b for a,b,v in matches};tp=len(matches);fp=sum(known[j] and j not in matched_pred for j in range(len(ps)));fn=len(self.refs)-tp;den=tp+.5*fp+.5*fn;unknown_unmatched=sum(not known[j] and j not in matched_pred for j in range(len(ps)))
        q=quality(self.refs,ps,self.n);vec=np.array([r['best_IoU'] for r in q]);m=self.compact
        metrics={'Mean_Best_IoU':float(vec[m].mean()) if m.any() else np.nan,'exact':float((vec[m]>=1-1e-12).mean()) if m.any() else np.nan,'near_0_8':float((vec[m]>=.8-1e-12).mean()) if m.any() else np.nan,'censored_PQ':sum(v for a,b,v in matches)/den if den else np.nan,'censored_RQ':tp/den if den else np.nan,'SQ':np.mean([v for a,b,v in matches]) if matches else 0.,'matched_object_precision':tp/(tp+fp) if tp+fp else np.nan,'matched_object_recall':tp/len(self.refs) if self.refs else np.nan,'TP':tp,'FP_known':fp,'FN':fn,'unmatched_unknown_objects':unknown_unmatched,'predicted_regions':len(ps),'reference_regions':len(self.refs),'compact_references':int(m.sum()),'compact_mean_coverage':np.mean([q[i]['coverage'] for i in np.flatnonzero(m)]),'compact_mean_purity':np.mean([q[i]['purity'] for i in np.flatnonzero(m)]),'compact_fragment_count':np.mean([q[i]['split_region_count'] for i in np.flatnonzero(m)])}
        small=sizes<=10;small_tp=sum(small[b] for a,b,v in matches);small_fp=sum(small[j] and known[j] and j not in matched_pred for j in range(len(ps)));metrics['matched_compact_prediction_precision']=small_tp/(small_tp+small_fp) if small_tp+small_fp else np.nan
        small_exact=sum(small[b] and v>=1-1e-12 for a,b,v in matches);small_known=sum(small&known);metrics['known_compact_exact_prediction_precision']=small_exact/small_known if small_known else np.nan
        for lo,hi in [(1,2),(3,5),(6,10)]+[(k,k) for k in range(1,11)]:
            band=(self.sizes>=lo)&(self.sizes<=hi);name=f'size{lo}_{hi}' if lo!=hi else f'size{lo}';metrics[name+'_Mean']=float(vec[band].mean()) if band.any() else np.nan;metrics[name+'_exact']=float((vec[band]>=1-1e-12).mean()) if band.any() else np.nan
        metrics.update(information_metrics(self.truth,labels))
        macro_known=~np.isin(np.char.lower(self.truth),list(MACRO_BAD));mt=self.truth[macro_known];mp=labels[macro_known]
        metrics['AMI']=float(adjusted_mutual_info_score(mt,mp,average_method='arithmetic')) if len(np.unique(mt))>1 and len(np.unique(mp))>1 else np.nan
        if details:return metrics,vec,q,ps,matches,known
        return metrics
