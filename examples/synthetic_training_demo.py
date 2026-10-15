"""Create artificial multi-map training inputs, not experimental evidence.

Usage: python examples/synthetic_training_demo.py --output synthetic_training
Then use the printed manifest with radarst train-inputs. All labels and counts
are generated at run time; no teacher decisions or fitted models are supplied.
"""
import argparse
from pathlib import Path


def generate(output):
    import numpy as np
    import pandas as pd
    import yaml
    from radarst.config import new_output
    from radarst.synthetic import generate as generate_section
    out=new_output(output)
    samples=[]
    for n in range(3):
        dataset=f'artificial_section_{n}'
        folder=out/dataset
        generate_section(folder)
        coordinates=pd.read_csv(folder/'coordinates.csv')
        ids=coordinates.spot_id.tolist()
        x=coordinates.x.to_numpy(int);y=coordinates.y.to_numpy(int)
        maps=[]
        for i in range(14):
            # Staggered broad artificial domains; no truth is consulted here.
            labels=((x+i%3)//3)+10*((y+(i//3)%3)//3)
            path=folder/f'workflow_{i:02d}.csv'
            pd.DataFrame(dict(spot_id=ids,pred_label=['domain_'+str(k) for k in labels])).to_csv(path,index=False)
            maps.append(dict(workflow=f'workflow_{i:02d}',labels=str(path.relative_to(out))))
        ref=pd.DataFrame(dict(spot_id=ids,truth_label='UNASSIGNED',truth_component_id=''))
        ref.loc[ref.spot_id.isin(ids[:3]),['truth_label','truth_component_id']]=['artificial_component_a',dataset+'__a']
        ref.loc[ref.spot_id.eq(ids[40]),['truth_label','truth_component_id']]=['artificial_component_b',dataset+'__b']
        ref.to_csv(folder/'reference_components.csv',index=False)
        samples.append(dict(dataset=dataset,cohort='development',
            expression=str((folder/'filtered_feature_bc_matrix.h5').relative_to(out)),
            coordinates=str((folder/'coordinates.csv').relative_to(out)),
            reference=str((folder/'reference_components.csv').relative_to(out)),upstream=maps))
    target=out/'manifest.yaml'
    target.write_text(yaml.safe_dump(dict(samples=samples),sort_keys=False))
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    generate(p.parse_args().output)
