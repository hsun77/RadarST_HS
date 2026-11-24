"""Frozen utility model plus properly routed full-map event-change predictor."""
from event_round2 import *
from object_evidence import gene_evidence,assemble
from fast_infer_round import checked
from quotient import reconstruct
from independent_evaluator import regions

def infer(ds,wf,utility,event,verifier,outroot=None,cold=None):
    out=(outroot or R2/'predictions')/ds/wf
    if (out/'PREDICTION_GATE.json').exists():return
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    if cold is None:
        c=load_map(ds,wf);genes=gene_evidence(c,ds);anchor=EX/'controls'/ds/wf/'CLEAN_V1';ag=js(anchor/'PREDICTION_GATE.json');verifier.check(anchor/'PREDICTION_GATE.json')
        for f,h in ag['files'].items():verifier.check(anchor/f,h)
        S0={c['lookup'][cid] for cid in pd.read_parquet(anchor/'INSTANCE_REGISTRY.parquet').candidate_id}
    else:c,genes,S0=cold
    n=len(c['a']['spots']);ids=c['ids'];members=c['members'];base=c['a']['base'];legal=[]
    for m in members:
        s=set(m);found={int(m[0])};todo=list(found)
        while todo:
            for j in c['adj'][todo.pop()]:
                if j in s and j not in found:found.add(j);todo.append(j)
        legal.append(found==s and 1<=len(s)<=10)
    legal=np.asarray(legal,bool);indices=np.flatnonzero(legal).tolist();cache={};methods=[]
    assert utility['columns']==event['columns']
    for policy in ['E2_EVENT_CREDIT']:
        S=S0.copy();seen=set();credit=0.;trace=[]
        for rnd in range(6):
            key=tuple(sorted(S))
            if key in seen:break
            seen.add(key)
            if key not in cache:
                ops=[int(i in S) for i in indices];dyn,feet,conf=checked(c,S,indices,ops,True);x=c['x'].iloc[indices].reset_index(drop=True).copy()
                for k in dyn:x[k]=dyn[k].to_numpy()
                for k in genes:x[k]=genes.iloc[indices][k].to_numpy()
                xx=x[utility['columns']].to_numpy(np.float32);assert np.isfinite(xx).all();pr=utility['model'].predict(xx)*utility['scale'];pr[:,:6]=np.maximum(pr[:,:6],0);pe=event['model'].predict(xx)*event['scale'];assert pe.shape==(len(indices),3);cache[key]=(pr,pe,feet,conf)
            pr,pe,feet,conf=cache[key];net=pr[:,:3]-pr[:,3:6];score=net.sum(1);events=pe.sum(1);allowed=(score>1e-12)&(net>=0).all(1)&(pr[:,6:9].sum(1)>=0)&(pr[:,9]>=0)
            if policy=='E1_LOCAL_EVENT':allowed&=events<=1e-12
            order=sorted(np.flatnonzero(allowed),key=lambda k:((events[k]>1e-12) if policy=='E2_EVENT_CREDIT' else False,-score[k],ids[indices[k]]));touched=set();changed=0
            for k in order:
                i=indices[k];foot=feet[k];closed=foot|{j for u in foot for j in c['adj'][u]}
                if touched&closed or not conf[k]<=S:continue
                if policy=='E2_EVENT_CREDIT' and events[k]>credit+1e-12:continue
                removal=i in S;old=conf[k];prior_credit=credit;S-=old
                if not removal:S.add(i)
                touched|=closed;changed+=1
                if policy=='E2_EVENT_CREDIT':credit-=float(events[k]);assert credit>=-1e-10
                trace.append(dict(round=rnd,candidate_id=ids[i],operation_remove=removal,removed_ids=';'.join(ids[j] for j in sorted(old)),predicted_net=float(score[k]),predicted_exact35=float(pr[k,7]),predicted_ARI=float(pr[k,9]),predicted_event_delta=float(events[k]),predicted_event_delta_1_2=float(pe[k,0]),predicted_event_delta_3_5=float(pe[k,1]),predicted_event_delta_6_10=float(pe[k,2]),expected_credit_before=prior_credit,expected_credit_after=credit))
            if not changed:break
        labels=labels_from(S,members,base);used=set()
        for i in S:assert legal[i] and not used&set(members[i]);used.update(members[i])
        source=out/policy;source.mkdir();pd.DataFrame(dict(spot_id=c['a']['spots'],refined=labels,edited=np.isin(np.arange(n),list(used)))).to_csv(source/'FINAL_LABELS.csv',index=False);c['frame'].iloc[sorted(S)].to_parquet(source/'SELECTED_OBJECTS.parquet',index=False);pd.DataFrame(trace,columns=['round','candidate_id','operation_remove','removed_ids','predicted_net','predicted_exact35','predicted_ARI','predicted_event_delta','predicted_event_delta_1_2','predicted_event_delta_3_5','predicted_event_delta_6_10','expected_credit_before','expected_credit_after']).to_csv(source/'DECISIONS.csv',index=False)
        put(source/'SELECTION_GATE.json',dict(status='PASS',method=policy,outer=ds,training_datasets=utility['training_datasets'],utility_model_sha256=utility['sha256'],event_model_sha256=event['sha256'],actual_modified_fraction=len(used)/n,selected_count=len(S),rounds=rnd+1,expected_final_event_credit=credit,no_reference_read=True,no_true_risk_or_non_decrease_guarantee=True,composition=c['frame'].iloc[sorted(S)].groupby(['proposal_type','n_spots']).size().reset_index(name='count').to_dict('records'),seconds_cumulative=time.time()-start));methods.append(policy)
        qname=policy+'__Q2_PAIR_OPT';qout=out/qname;qout.mkdir();qlabels,info=reconstruct(labels,base,c['a']['edge'],c['adj'],c['a']['pcs'],molecular=False,pair_optimize=True);pd.DataFrame(dict(spot_id=c['a']['spots'],refined=qlabels)).to_csv(qout/'FINAL_LABELS.csv',index=False);registry=[]
        selected={tuple(sorted(str(c['a']['spots'][k]) for k in members[i])):ids[i] for i in S}
        for k,m in enumerate(regions(qlabels,c['adj'])):
            spots=tuple(sorted(str(c['a']['spots'][u]) for u in m));registry.append(dict(instance_id=hashlib.sha256((ds+'|'+';'.join(spots)).encode()).hexdigest(),region_index=k,domain_label=int(qlabels[min(m)]),n_spots=len(m),member_spots=';'.join(spots),source_selected_candidate_id=selected.get(spots,'')))
        pd.DataFrame(registry).to_parquet(qout/'REGION_IDENTITY.parquet',index=False);put(qout/'QUOTIENT_GATE.json',dict(status='PASS',source_prediction_sha256=sha(source/'FINAL_LABELS.csv'),actual_modified_fraction=len(used)/n,no_reference_read=True,**info));methods.append(qname);print('EVENT_POLICY_PREDICTED',ds,wf,policy,len(S),time.time()-start,flush=True)
    put(out/'PREDICTION_GATE.json',dict(status='PASS',dataset=ds,workflow=wf,methods=methods,no_reference_read=True,protocol_sha256=sha(R2/'PROTOCOL.json'),files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},utc=utc()))

def bundles(ds):
    p=ROOT/'models'/ds/'B1_ACTUAL_OBJECT';u=js(p/'MODEL_GATE.json');assert ds not in u['training_datasets'] and u['strict_inner_state_exclusion'];assert sha(p/'MODEL.joblib')==u['model_sha256'];utility=joblib.load(p/'MODEL.joblib');utility['sha256']=u['model_sha256'];p=R2/'models'/ds;g=js(p/'MODEL_GATE.json');assert ds not in g['training_datasets'];assert sha(p/'EVENT_MODEL.joblib')==g['model_sha256'];event=joblib.load(p/'EVENT_MODEL.joblib');event['sha256']=g['model_sha256'];return utility,event

