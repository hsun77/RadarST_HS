"""Exact full-context region update for add, restore and legal exchange targets."""
import numpy as np
from exact_engine import Counterfactual,partition,indices,iouvec
class InterventionEngine(Counterfactual):
    def apply(self,after,check=False):
        after=np.asarray(after);changed=np.flatnonzero(after!=self.base).tolist()
        if not changed:return self.baseline.copy()
        D=set(changed);affected={self.vertex_region[i] for i in changed}
        # Only changed vertices can introduce a new link to an old region. Keep
        # the ENTIRE old region, including arbitrarily distant boundary context.
        for i in changed:
            for j in self.adj[i]:
                if j not in D and after[i]==self.base[j]:affected.add(self.vertex_region[j])
        union=0
        for r in affected:union|=self.regions[r]
        new=partition(after,self.adj,indices(union));values=self.baseline.copy();truth_affected={j for r in affected for j in self.affected_truth[r]}
        for j in truth_affected:
            unchanged=next((self.quality[j,r] for r in self.rank[j] if r not in affected),0.);t=self.truth[j];added=max(((r&t).bit_count()/(r|t).bit_count() for r in new if r&t),default=0.);values[j]=max(unchanged,added)
        if check:
            full=partition(after,self.adj);assert set(full)==set([r for k,r in enumerate(self.regions) if k not in affected]+new);assert np.allclose(values,iouvec(full,self.truth),rtol=0,atol=1e-12)
        return values
