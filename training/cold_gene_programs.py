"""Raw prepared expression -> frozen shared programs, never old gene arrays."""
from instance_quality_schema import *
import argparse,numpy as np,pandas as pd
sys.path.insert(0,str(ROOT/'deploy_modules'))
from gene_reader import read10x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--workflow',default='GraphST');ap.add_argument('--run',default='cold_v3');a=ap.parse_args();qguard('inference');ds=a.dataset;samples=pd.read_csv(OLD/'frozen_assets/clean_runtime/manifests/SAMPLES.csv',keep_default_na=False);s=next(s for s in samples.itertuples() if s.dataset==ds);front=ROOT/'cold_runs'/a.run/ds/a.workflow;fg=js(front/'COLD_FRONTEND_GATE.json');assert fg['status']=='PASS' and fg['no_historical_candidates_scores_state_or_accepted_cache_read'];assert sha(front/'MAP.npz')==fg['files']['MAP.npz'];assert sha(s.cfar_expression)==fg['raw_input_hashes'][s.cfar_expression];p=R3/'encoders_union'/ds;eg=js(p/'ENCODER_GATE.json');assert ds not in eg['training_datasets'] and sha(p/'GENE_PROGRAM_ENCODER.npz')==eg['encoder_sha256']
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3];write=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if not write and any(path.is_relative_to(q) for q in [ROOT/'inputs',ROOT/'features',R3/'features',R3/'features_union',EX/'features',EX/'inputs',EX/'controls',AF,OLD/'runs']):raise RuntimeError('GENE_CACHE_IN_RAW_PROGRAMS '+str(path))
    sys.addaudithook(hook)
    with np.load(front/'MAP.npz',allow_pickle=False) as z:spots=z['spots']
    with np.load(p/'GENE_PROGRAM_ENCODER.npz',allow_pickle=False) as z:genes=z['genes'];mean=z['mean'];basis=z['basis'];scale=z['scale']
    raw,provided,meta=read10x(Path(s.cfar_expression),spots,normalize=False)
    if meta['integer_like_nonzero_fraction']>=1-1e-6:
        total=np.asarray(raw.sum(1)).ravel();raw=raw.multiply((1e4/np.maximum(total,1))[:,None]).tocsr();raw.data=np.log1p(raw.data)
    x=raw.toarray().astype(np.float32);z=np.clip((x-x.mean(0))/np.maximum(x.std(0),1e-4),-6,6);lookup={str(g):i for i,g in enumerate(genes)};keep=[k for k,g in enumerate(provided) if str(g) in lookup];ix=[lookup[str(provided[k])] for k in keep];out=np.empty((len(spots),64),np.float32);offset=-(mean.astype(np.float64)@basis.astype(np.float64))
    for start in range(0,len(spots),256):
        stop=min(start+256,len(spots));v=np.asarray(z[start:stop,keep],np.float64);out[start:stop]=((v@basis[ix].astype(np.float64)+offset)/scale).astype(np.float32)
    dst=R3/'cold_programs'/a.run/ds;dst.mkdir(parents=True,exist_ok=False);np.save(dst/'PROGRAMS.npy',out,allow_pickle=False);np.save(dst/'SPOTS.npy',spots,allow_pickle=False);put(dst/'RAW_PROGRAM_GATE.json',dict(status='PASS',outer=ds,raw_expression_sha256=sha(s.cfar_expression),encoder_sha256=eg['encoder_sha256'],gene_fraction=len(keep)/len(genes),no_historical_gene_features_or_candidate_cache_read=True,no_reference_read=True,files={n:sha(dst/n) for n in ['PROGRAMS.npy','SPOTS.npy']},utc=utc()));print('RAW_GENE_PROGRAMS_COMPLETE',ds,flush=True)

if __name__=='__main__':main()
