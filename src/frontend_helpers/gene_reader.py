import numpy as np,h5py
from scipy import sparse
def strings(x):return np.asarray([v.decode('utf8') if isinstance(v,bytes) else str(v) for v in x])
def read10x(path,spots,normalize=True):
    with h5py.File(path,'r') as f:
        g=f['matrix'];shape=tuple(g['shape'][()]);x=sparse.csc_matrix((g['data'][()].astype(np.float32),g['indices'][()],g['indptr'][()]),shape=shape).T.tocsr();bc=strings(g['barcodes'][()]);feat=g['features'];symbols=strings(feat['name'][()]);ids=strings(feat['id'][()]);kind=strings(feat['feature_type'][()]) if 'feature_type' in feat else np.repeat('Gene Expression',len(symbols))
    valid=kind=='Gene Expression';x=x[:,valid];symbols=symbols[valid];ids=ids[valid]
    integer_fraction=float(np.mean(np.abs(x.data-np.rint(x.data))<1e-5)) if len(x.data) else 1.
    assert len(bc)==len(set(bc)),path
    lookup={s:i for i,s in enumerate(bc)};alignment='exact barcode'
    if not set(spots)<=set(lookup):
        canon=lambda s:s[:-2] if s.endswith('-1') else s
        clean=[canon(s) for s in bc];want=[canon(str(s)) for s in spots]
        assert len(set(clean))==len(clean) and len(set(want))==len(want) and set(want)<=set(clean),'Non-bijective expression barcode alignment'
        lookup={s:i for i,s in enumerate(clean)};indices=[lookup[s] for s in want];alignment='verified bijective optional -1 suffix normalization'
    else:indices=[lookup[str(s)] for s in spots]
    x=x[indices].tocsr();assert np.isfinite(x.data).all() and (x.data>=0).all()
    genes=np.asarray([s.strip().upper() for s in symbols]);unique=sorted(set(genes)-{''});gi={g:i for i,g in enumerate(unique)}
    rows=np.array([i for i,g in enumerate(genes) if g in gi]);columns=np.array([gi[genes[i]] for i in rows]);agg=sparse.csr_matrix((np.ones(len(rows),np.float32),(rows,columns)),shape=(len(genes),len(unique)))
    x=(x@agg).tocsr();totals=np.asarray(x.sum(1)).ravel()
    if normalize:x=x.multiply((1e4/np.maximum(totals,1))[:,None]).tocsr();x.data=np.log1p(x.data).astype(np.float32)
    return x,np.asarray(unique),{'alignment':alignment,'source_genes':len(genes),'unique_symbol_genes':len(unique),'duplicate_symbols_aggregated':len(genes)-len(unique),'zero_library_spots':int((totals<=0).sum()),'integer_like_nonzero_fraction':integer_fraction,'normalization':'CP10k+log1p (initial audit v0 only)' if normalize else 'provided nonnegative prepared expression preserved; count origin not assumed','source_gene_ids':ids.tolist(),'source_symbols':symbols.tolist()}
