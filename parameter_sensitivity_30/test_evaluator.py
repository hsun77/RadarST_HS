"""Engineering parity on synthetic data only, before reference evaluation."""
import evaluate_sensitivity as e
import numpy as np
from object_metrics import ObjectEvaluator
from object_evidence import adjacency
edge=np.array([(i,i+1) for i in range(29)],int)
truth=np.repeat(np.array(['a','unknown','b','c','d','e']),5)
ev=ObjectEvaluator(truth,adjacency(30,edge));rng=np.random.default_rng(122);err=0.
for k in range(80):
 labels=rng.integers(0,6,size=30);q,c,p,s=e.quality(labels,edge,ev.refs,ev.sizes)
 m,v,rows,*_=ev.evaluate(labels,True);err=max(err,float(abs(q-v).max()))
 assert np.allclose(q,v,atol=1e-12,rtol=0)
 assert np.allclose(c,[r['coverage'] for r in rows])
 assert np.allclose(p,[r['purity'] for r in rows])
 assert np.array_equal(s,[r['split_region_count'] for r in rows])
edge=np.array([(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(7,8),(8,9),(4,10),(10,11),(11,12),(12,13),(13,14),(14,15)])
truth=np.array(['T']*5+['unknown']*11);labels=np.array([0,0,1,2,2,0,0,0,0,0,2,2,2,2,2,2]);ev=ObjectEvaluator(truth,adjacency(16,edge));q,c,p,s=e.quality(labels,edge,ev.refs,ev.sizes)
assert np.allclose(q,[.2]) and np.allclose(c,[.4]) and np.allclose(p,[2/7]) and np.array_equal(s,[3])
m,v,rows,*_=ev.evaluate(labels,True);assert np.array_equal(q,v) and np.allclose(p,[r['purity'] for r in rows])
e.put(e.W/'qa/EVALUATOR_ENGINEERING_TESTS.json',dict(synthetic_labelings=81,explicit_equal_IoU_different_purity_case=True,tie_break='prediction component with lowest member index, matching frozen region ordering',independent_explicit_set_max_error=err,real_reference_files_read=False))
print('SYNTHETIC_EVALUATOR_PARITY',81,err)
