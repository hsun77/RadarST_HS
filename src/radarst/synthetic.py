"""Deterministic, fully artificial input generation, not research data."""
from pathlib import Path


def generate(output):
    import numpy as np
    import pandas as pd
    import h5py
    from scipy import sparse
    from .config import new_output
    out=new_output(output)
    rng=np.random.default_rng(1729)
    xy=np.array([(x,y) for y in range(9) for x in range(9)],dtype=float)
    ids=[f"synthetic_{i:03d}" for i in range(len(xy))]
    counts=rng.poisson(3,size=(len(ids),96))
    counts[(xy[:,0]>3)&(xy[:,0]<6),:12]+=10
    x=sparse.csc_matrix(counts.T)
    with h5py.File(out/"filtered_feature_bc_matrix.h5","w") as h:
        g=h.create_group("matrix")
        for name,value in [("data",x.data),("indices",x.indices),("indptr",x.indptr),("shape",x.shape)]:g.create_dataset(name,data=value)
        g.create_dataset("barcodes",data=np.asarray(ids,dtype="S"))
        f=g.create_group("features")
        genes=np.asarray([f"SYNTHETIC_GENE_{i:03d}" for i in range(96)],dtype="S")
        f.create_dataset("name",data=genes);f.create_dataset("id",data=genes)
        f.create_dataset("feature_type",data=np.asarray(["Gene Expression"]*96,dtype="S"))
        f.create_dataset("genome",data=np.asarray(["synthetic"]*96,dtype="S"))
    pd.DataFrame({"spot_id":ids,"x":xy[:,0],"y":xy[:,1]}).to_csv(out/"coordinates.csv",index=False)
    pd.DataFrame({"spot_id":ids,"pred_label":np.where(xy[:,0]<4,"synthetic_left","synthetic_right")}).to_csv(out/"upstream_labels.csv",index=False)
    pd.DataFrame({"spot_id":ids,"truth_label":np.where((xy[:,0]==4)&(xy[:,1]==4),"synthetic_compact","synthetic_background")}).to_csv(out/"reference.csv",index=False)
    (out/"SYNTHETIC.txt").write_text("Entirely artificial software-test data; not a biological sample. No fitted model is supplied.\n")
    print(out)
