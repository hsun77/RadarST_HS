"""Round2: directly align risk supervision to the actual upstream loss-event metric."""
from common import *
from object_evidence import load_map,labels_from
from object_metrics import ObjectEvaluator
from intervention import InterventionEngine
from exact_engine import maskof
from prepare_round import Y
import argparse,numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesRegressor
R2=ROOT/'round2_event_alignment'
EY=[f'event_delta_{lo}_{hi}' for lo,hi in [(1,2),(3,5),(6,10)]]
KEY=['dataset','workflow','candidate_id','operation_remove']

def rguard(role,T=None):
    guard(role,T)
    if role in ['training','inference','observable']:
        def hook(event,args):
            if event=='open' and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3];write=(isinstance(mode,str) and any(s in mode for s in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
                if not write and p.is_relative_to(R2/'evaluation'):raise RuntimeError('ROUND2_ENDPOINT_IN_LEARNING '+str(p))
                if not write and p.is_relative_to(R2/'targets'):
                    if role in ['inference','observable']:raise RuntimeError('EVENT_TARGET_IN_DEPLOYMENT '+str(p))
                    pp=p.relative_to(R2/'targets').parts
                    if pp and pp[0]=='upstream' and len(pp)>1 and pp[1]==T:raise RuntimeError('OUTER_EVENT_TARGET_READ '+str(p))
                    if pp and pp[0]=='nested' and len(pp)>1 and pp[1]!=T:raise RuntimeError('OTHER_FOLD_EVENT_TARGET_READ '+str(p))
        sys.addaudithook(hook)

def event_delta(before,after,up,sizes):
    change=(after<up-1e-12).astype(np.int32)-(before<up-1e-12).astype(np.int32)
    return [int(change[(sizes>=lo)&(sizes<=hi)].sum()) for lo,hi in [(1,2),(3,5),(6,10)]]

def prepare_one(J,T=None):
    out=R2/'targets'/('upstream' if T is None else 'nested')
    if T:out=out/T
    out=out/J;gatefile=out/'EVENT_TARGET_GATE.json'
    if gatefile.exists():
        g=js(gatefile);assert sha(out/'EVENT_TARGETS.parquet')==g['target_sha256'];return
    src=ROOT/'targets'/J/'UPSTREAM.parquet' if T is None else ROOT/'targets'/T/'nested'/J/'CONTEXT.parquet';sg=js(src.parent/('UPSTREAM_GATE.json' if T is None else 'CONTEXT_GATE.json'));assert sha(src)==sg['target_sha256'];tab=pd.read_parquet(src);assert set(tab.dataset)=={J};assert not tab.duplicated(KEY).any();rows=[];checks=[]
    for wf,t in tab.groupby('workflow',sort=True):
        c=load_map(J,wf);S=set()
        if T:
            route=sg['routes'][wf];assert set(route['excluded'])=={T,J} and set(route['training_datasets']).isdisjoint({T,J});state=AF/'onpolicy_states'/'__'.join(sorted([T,J]))/J/wf;stateg=js(state/'STATE_GATE.json');assert sha(state/'STATE_GATE.json')==route['state_gate_sha256'];assert set(stateg['excluded'])=={T,J};selected=pd.read_csv(state/'SELECTED_OBJECTS.csv');assert sha(state/'SELECTED_OBJECTS.csv')==stateg['files']['SELECTED_OBJECTS.csv'];S={c['lookup'][cid] for cid in selected.candidate_id}
        reg=next(r for r in pairs() if r['dataset']==J and r['workflow']==wf);assert sha(reg['reference'])==reg['reference_sha256'];truth=pd.read_csv(reg['reference'],dtype=str,keep_default_na=False).set_index('spot_id').truth_label.reindex(c['a']['spots']);assert truth.notna().all();ev=ObjectEvaluator(truth.to_numpy(),c['adj']);refs=[maskof(s) for s in ev.refs];up=InterventionEngine(c['a']['base'],c['adj'],refs).baseline;cur=labels_from(S,c['members'],c['a']['base']);engine=InterventionEngine(cur,c['adj'],refs);owner=np.full(len(cur),-1,int)
        for i in S:owner[c['members'][i]]=i
        seenops=set()
        for r in t.itertuples():
            i=c['lookup'][r.candidate_id];op=bool(r.operation_remove);cf={i} if op else set(owner[c['members'][i]])-{-1};lab=cur.copy()
            for j in cf:lab[c['members'][j]]=c['a']['base'][c['members'][j]]
            if not op:lab[c['members'][i]]=int(cur.max())+1
            q=engine.apply(lab,check=op not in seenops);ys=event_delta(engine.baseline,q,up,ev.sizes)
            if op not in seenops:
                # Separate full connected-partition evaluator checks the count target.
                before2=ev.evaluate(cur,True)[1];after2=ev.evaluate(lab,True)[1];up2=ev.evaluate(c['a']['base'],True)[1];assert ys==event_delta(before2,after2,up2,ev.sizes)
                delta=q-engine.baseline
                for band,(lo,hi) in enumerate([(1,2),(3,5),(6,10)]):
                    m=(ev.sizes>=lo)&(ev.sizes<=hi);assert abs(float(np.maximum(delta[m],0).sum())-getattr(r,f'y_gain_{lo}_{hi}'))<1e-10;assert abs(float(np.maximum(-delta[m],0).sum())-getattr(r,f'y_harm_{lo}_{hi}'))<1e-10
                checks.append(dict(workflow=wf,operation_remove=op,full_partition_parity=True,original_gain_harm_target_parity=True));seenops.add(op)
            rows.append(dict(dataset=J,workflow=wf,candidate_id=r.candidate_id,operation_remove=int(op),**dict(zip(EY,ys))))
        print('EVENT_TARGET_WORKFLOW',T or 'UPSTREAM',J,wf,len(t),flush=True)
    out.mkdir(parents=True,exist_ok=False);df=pd.DataFrame(rows);assert len(df)==len(tab) and not df.duplicated(KEY).any();df.to_parquet(out/'EVENT_TARGETS.parquet',index=False);put(gatefile,dict(status='PASS',outer=T,training_sample=J,rows=len(df),source_original_target_sha256=sha(src),source_original_gate_sha256=sha(src.parent/('UPSTREAM_GATE.json' if T is None else 'CONTEXT_GATE.json')),strict_outer_and_receiver_exclusion=T is not None,no_supervised_generator=T is None,full_partition_checks=checks,target_sha256=sha(out/'EVENT_TARGETS.parquet'),utc=utc()))

def prepare(T=None,J=None):
    rguard('training' if T else 'target',T)
    if T:
        for j in sorted({r['dataset'] for r in pairs()}-{T}):prepare_one(j,T)
    else:prepare_one(J)

def fit(T,threads):
    rguard('training',T);lock=js(R2/'IMPLEMENTATION_LOCK_PRE_FIT.json')
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,('EVENT_IMPLEMENTATION_CHANGED',rel)
    assert js(R2/'audits/ENGINEERING_TESTS.json')['status']=='PASS'
    train=sorted({r['dataset'] for r in pairs()}-{T});parts=[];sources={};physical={}
    for J in train:
        for old,new in [(ROOT/'targets'/J/'UPSTREAM.parquet',R2/'targets/upstream'/J/'EVENT_TARGETS.parquet'),(ROOT/'targets'/T/'nested'/J/'CONTEXT.parquet',R2/'targets/nested'/T/J/'EVENT_TARGETS.parquet')]:
            gate=js(new.parent/'EVENT_TARGET_GATE.json');assert sha(new)==gate['target_sha256'] and sha(old)==gate['source_original_target_sha256'];t=pd.read_parquet(old);ev=pd.read_parquet(new);t=t.merge(ev,on=KEY,validate='one_to_one');assert set(t.dataset)=={J};parts.append(t);sources[str(new)]=sha(new);sources[str(old)]=sha(old)
        ids=pd.read_parquet(ROOT/'inputs'/J/'GraphST'/'CANDIDATES.parquet');physical.update({cid:hashlib.sha256(';'.join(sorted(m.split(';'))).encode()).hexdigest() for cid,m in zip(ids.candidate_id,ids.member_spots)})
    tab=pd.concat(parts,ignore_index=True);good=np.isfinite(tab[Y].to_numpy()).all(1);tab=tab.loc[good].reset_index(drop=True);y=tab[EY].to_numpy(np.float32);assert np.isfinite(y).all();ph=tab.candidate_id.map(physical);assert ph.notna().all();keys=tab[['dataset','regime','workflow','operation_remove']].astype(str).agg('|'.join,axis=1)+'|'+ph;cnt=keys.value_counts();w=keys.map(lambda k:1/cnt[k]).to_numpy()
    for cols in [['dataset','regime','workflow'],['dataset','regime'],['dataset']]:
        k=tab[cols].astype(str).agg('|'.join,axis=1);w/=pd.Series(w).groupby(k.to_numpy()).transform('sum').to_numpy()
    w*=len(w)/w.sum();mean=np.average(y,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((y-mean)**2,axis=0,weights=w)),.01);original=js(ROOT/'models'/T/'B1_ACTUAL_OBJECT/MODEL_GATE.json');columns=original['columns'];assert len(tab)==original['rows'];x=tab[columns].to_numpy(np.float32);assert np.isfinite(x).all();model=ExtraTreesRegressor(n_estimators=256,min_samples_leaf=8,max_features=.7,random_state=20261104,n_jobs=threads,bootstrap=False);model.fit(x,y/scale,sample_weight=w);model.n_jobs=1;out=R2/'models'/T;out.mkdir(parents=True,exist_ok=False);joblib.dump(dict(model=model,columns=columns,targets=EY,scale=scale,heldout=T,training_datasets=train),out/'EVENT_MODEL.joblib',compress=3);put(out/'MODEL_GATE.json',dict(status='PASS',outer=T,training_datasets=train,rows=len(tab),columns=columns,original_utility_model_kept_frozen=True,source_hashes=sources,model_sha256=sha(out/'EVENT_MODEL.joblib'),protocol_sha256=sha(R2/'PROTOCOL.json'),utc=utc()));print('EVENT_MODEL_FITTED',T,len(tab),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--upstream');ap.add_argument('--outer');ap.add_argument('--fit');ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();assert sum(bool(x) for x in [a.upstream,a.outer,a.fit])==1
    fit(a.fit,a.threads) if a.fit else prepare(a.outer,a.upstream)
