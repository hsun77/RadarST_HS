from common import *
import infer_round
from fast_geometry import dynamic_features as fast
from object_evidence import dynamic_features as slow
import numpy as np
def checked(c,S,which,operations=None,with_residual=True):
 result=fast(c,S,which,operations,with_residual)
 # Deterministic engineering checks in every real scoring state; no references.
 take=sorted(set(np.linspace(0,len(which)-1,min(8,len(which)),dtype=int).tolist()))
 if take:
  op=None if operations is None else np.asarray(operations)[take];ref=slow(c,S,[which[k] for k in take],op,with_residual);pdiff=result[0].iloc[take].to_numpy(np.float32);assert np.array_equal(pdiff,ref[0].to_numpy(np.float32)),'ONLINE_FEATURE_ENGINE_MISMATCH'
  if with_residual:assert [result[1][k] for k in take]==ref[1] and [result[2][k] for k in take]==ref[2]
 return result
if __name__=='__main__':
 gate=js(ROOT/'audits/FAST_GEOMETRY_PARITY.json');assert gate['status']=='PASS'
 assert sha(ROOT/'code/fast_geometry.py')==gate['source_files']['new_engine']
 assert sha(ROOT/'code/object_evidence.py')==gate['source_files']['locked_original_engine']
 assert js(ROOT/'audits/FAST_TRAINED_DECISION_PARITY.json')['status']=='PASS'
 assert js(ROOT/'audits/ALL_OBSERVABLE_INPUTS_VERIFIED.json')['status']=='PASS'
 infer_round.dynamic_features=checked
 infer_round.main()
