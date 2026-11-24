"""Exact two-regime E2 FULL analogue, excluding receiver from policy generator."""
from bridge import *
import argparse

def prepare():
    guard('training');check_lock();bind()
    import numpy as np, pandas as pd
    from prepare_round import Y
    from event_round2 import EY,KEY
    train=sorted({r['dataset'] for r in pairs('development')});parts=[];physical={};sources={};routes=[]
    def check(p): sources[str(p)]=frozen(p)
    cols=None
    for J in train:
        original=SOURCE/'targets'/J/'UPSTREAM.parquet';ep=SOURCE/'round2_event_alignment/targets/upstream'/J/'EVENT_TARGETS.parquet'
        for p in [original,original.parent/'UPSTREAM_GATE.json',ep,ep.parent/'EVENT_TARGET_GATE.json']:check(p)
        eg=js(ep.parent/'EVENT_TARGET_GATE.json');ug=js(original.parent/'UPSTREAM_GATE.json')
        assert sha(original)==ug['target_sha256']==eg['source_original_target_sha256'] and sha(ep)==eg['target_sha256']
        a=pd.read_parquet(original);e=pd.read_parquet(ep);assert not a.duplicated(KEY).any() and not e.duplicated(KEY).any()
        u=a.merge(e,on=KEY,validate='one_to_one');assert len(u)==len(a)==len(e);parts.append(u)
        cp=SOURCE/'full_fit_targets'/J/'CONTEXT.parquet';gp=cp.parent/'FULL_CONTEXT_GATE.json'
        check(cp);check(gp);g=js(gp)
        assert g['status']=='PASS' and g['receiver_excluded_from_generator'] and not g['external_labels_or_data_read'] and sha(cp)==g['target_sha256']
        for wf,route in g['routes'].items():
            assert wf in ['GraphST','Seurat'] and set(route['generator_training_datasets'])==set(train)-{J} and route['heldout_receiver']==J
        t=pd.read_parquet(cp);assert set(t.dataset)=={J} and all(k in t for k in EY)
        parts.append(t);routes.append(dict(receiver=J,generator_training_samples=sorted(set(train)-{J}),workflows=sorted(g['routes'])))
        ip=SOURCE/'inputs'/J/'GraphST/CANDIDATES.parquet';check(ip);identity=pd.read_parquet(ip)
        for cid,m in zip(identity.candidate_id,identity.member_spots):
            h=hashlib.sha256(';'.join(sorted(m.split(';'))).encode()).hexdigest()
            assert cid not in physical or physical[cid]==h
            physical[cid]=h
        for rel in [f'models/{J}/B1_ACTUAL_OBJECT/MODEL_GATE.json',f'round2_event_alignment/models/{J}/MODEL_GATE.json']:
            p=SOURCE/rel;check(p);cg=js(p);assert set(cg['training_datasets'])==set(train)-{J}
            if cols is None:cols=cg['columns']
            assert cols==cg['columns'] and len(cols)==201
    tab=pd.concat(parts,ignore_index=True);valid=np.isfinite(tab[Y].to_numpy()).all(1);dropped=int((~valid).sum());tab=tab.loc[valid].reset_index(drop=True)
    assert set(tab.dataset)==set(train) and set(tab.regime)=={'upstream','onpolicy'}
    ph=tab.candidate_id.map(physical);assert ph.notna().all();keys=tab[['dataset','regime','workflow','operation_remove']].astype(str).agg('|'.join,axis=1)+'|'+ph
    cnt=keys.value_counts();w=keys.map(lambda k:1/cnt[k]).to_numpy()
    for cc in [['dataset','regime','workflow'],['dataset','regime'],['dataset']]:
        k=tab[cc].astype(str).agg('|'.join,axis=1);w/=pd.Series(w).groupby(k.to_numpy()).transform('sum').to_numpy()
    w*=len(w)/w.sum();assert np.isfinite(tab[cols+Y+EY].to_numpy(np.float64)).all()
    p=ROOT/'training';p.mkdir(exist_ok=False);tab.to_parquet(p/'TRAINING.parquet',index=False);np.save(p/'WEIGHTS.npy',w,allow_pickle=False)
    put(p/'TRAINING_GATE.json',dict(status='PASS',training_datasets=train,rows=len(tab),dropped_invalid_original_targets=dropped,columns=cols,utility_targets=Y,event_targets=EY,regime_rows=tab.groupby('regime').size().to_dict(),receiver_routes=routes,source_hashes=sources,table_sha256=sha(p/'TRAINING.parquet'),weights_sha256=sha(p/'WEIGHTS.npy'),no_external_data_read=True,utc=utc()))
    print('FULL_E2_TRAINING_PREPARED',len(tab),flush=True)

def fit(family):
    guard('training');check_lock();bind()
    import numpy as np,pandas as pd,joblib
    from sklearn.ensemble import ExtraTreesRegressor
    p=ROOT/'training';g=js(p/'TRAINING_GATE.json');assert sha(p/'TRAINING.parquet')==g['table_sha256'] and sha(p/'WEIGHTS.npy')==g['weights_sha256']
    t=pd.read_parquet(p/'TRAINING.parquet');w=np.load(p/'WEIGHTS.npy',allow_pickle=False);cols=g['columns'];targets=g['utility_targets' if family=='UTILITY' else 'event_targets']
    x=t[cols].to_numpy(np.float32);y=t[targets].to_numpy(np.float32);assert np.isfinite(x).all() and np.isfinite(y).all()
    scale=np.sqrt(np.average((y-np.average(y,axis=0,weights=w))**2,axis=0,weights=w))
    if family=='UTILITY':scale[:9]=np.maximum(scale[:9],.01);scale[9]=max(scale[9],1e-8)
    else:scale=np.maximum(scale,.01)
    model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=8,max_features=.7,n_jobs=4,random_state=20261104,bootstrap=False)
    start=time.time();model.fit(x,y/scale,sample_weight=w);model.n_jobs=1
    out=ROOT/'models'/family;out.mkdir(parents=True,exist_ok=False)
    joblib.dump(dict(model=model,columns=cols,targets=targets,scale=scale,heldout='EXTERNAL_COHORT',training_datasets=g['training_datasets']),out/'MODEL.joblib',compress=3)
    put(out/'MODEL_GATE.json',dict(status='PASS',family=family,training_datasets=g['training_datasets'],regime_rows=g['regime_rows'],rows=len(t),columns=cols,targets=targets,model_sha256=sha(out/'MODEL.joblib'),training_table_sha256=g['table_sha256'],protocol_sha256=sha(ROOT/'PROTOCOL_LOCK.json'),seconds=time.time()-start,no_external_data_or_labels_read=True,utc=utc()))
    print('FULL_E2_MODEL_FITTED',family,len(t),time.time()-start,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','UTILITY','EVENT']);a=ap.parse_args()
    prepare() if a.stage=='prepare' else fit(a.stage)
