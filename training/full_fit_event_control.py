"""FULL event learner, only after frozen round2 candidate selection."""
from event_round2 import *
from full_fit_candidate import full_guard,get_lock

def main():
    full_guard();rguard('training');lock=get_lock();assert lock['selection_record'] in ['round2_event_alignment/DECISION.json','round3_instance_quality/DECISION.json'];train=sorted({r['dataset'] for r in pairs()});parts=[];sources={};physical={}
    for J in train:
        original=ROOT/'targets'/J/'UPSTREAM.parquet';evp=R2/'targets/upstream'/J/'EVENT_TARGETS.parquet';g=js(evp.parent/'EVENT_TARGET_GATE.json');assert sha(evp)==g['target_sha256'] and sha(original)==g['source_original_target_sha256'];t=pd.read_parquet(original).merge(pd.read_parquet(evp),on=KEY,validate='one_to_one');parts.append(t);sources[str(original)]=sha(original);sources[str(evp)]=sha(evp)
        p=ROOT/'full_fit_targets'/J/'CONTEXT.parquet';g=js(p.parent/'FULL_CONTEXT_GATE.json');assert g['receiver_excluded_from_generator'] and not g['external_labels_or_data_read'] and sha(p)==g['target_sha256'];t=pd.read_parquet(p);assert set(t.dataset)=={J} and all(k in t for k in EY);parts.append(t);sources[str(p)]=sha(p)
        ident=pd.read_parquet(ROOT/'inputs'/J/'GraphST'/'CANDIDATES.parquet');physical.update({cid:hashlib.sha256(';'.join(sorted(m.split(';'))).encode()).hexdigest() for cid,m in zip(ident.candidate_id,ident.member_spots)})
    tab=pd.concat(parts,ignore_index=True);good=np.isfinite(tab[Y].to_numpy()).all(1);tab=tab.loc[good].reset_index(drop=True);y=tab[EY].to_numpy(np.float32);assert np.isfinite(y).all();ph=tab.candidate_id.map(physical);assert ph.notna().all();keys=tab[['dataset','regime','workflow','operation_remove']].astype(str).agg('|'.join,axis=1)+'|'+ph;cnt=keys.value_counts();w=keys.map(lambda k:1/cnt[k]).to_numpy()
    for cols in [['dataset','regime','workflow'],['dataset','regime'],['dataset']]:
        k=tab[cols].astype(str).agg('|'.join,axis=1);w/=pd.Series(w).groupby(k.to_numpy()).transform('sum').to_numpy()
    w*=len(w)/w.sum();mean=np.average(y,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((y-mean)**2,axis=0,weights=w)),.01);cols=js(R2/'models'/train[0]/'MODEL_GATE.json')['columns'];assert len(cols)==201
    for J in train:assert js(R2/'models'/J/'MODEL_GATE.json')['columns']==cols
    x=tab[cols].to_numpy(np.float32);assert np.isfinite(x).all();model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=8,max_features=.7,random_state=20261104,n_jobs=4,bootstrap=False);model.fit(x,y/scale,sample_weight=w);model.n_jobs=1;out=ROOT/'full_fit_models/EVENT_CONTROL';out.mkdir(parents=True,exist_ok=False);joblib.dump(dict(model=model,columns=cols,targets=EY,scale=scale,heldout='EXTERNAL_COHORT',training_datasets=train),out/'EVENT_MODEL.joblib',compress=3);put(out/'MODEL_GATE.json',dict(status='PASS',training_datasets=train,rows=len(tab),columns=cols,targets=EY,source_hashes=sources,model_sha256=sha(out/'EVENT_MODEL.joblib'),external_data_or_labels_read=False,selection_fixed_before_external=True,utc=utc()));print('FULL_EVENT_MODEL_FITTED',len(tab),flush=True)

if __name__=='__main__':main()
