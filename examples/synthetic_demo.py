"""Generate artificial input and run the real feature front end, without weights."""
import argparse
from pathlib import Path
from radarst.cli import main

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.output).resolve()
main(['synthetic','--output',str(root/'inputs')])
main(['prepare-features','--expression',str(root/'inputs/filtered_feature_bc_matrix.h5'),
      '--coordinates',str(root/'inputs/coordinates.csv'),
      '--upstream-labels',str(root/'inputs/upstream_labels.csv'),
      '--dataset','synthetic','--workflow','example','--output',str(root/'features')])
