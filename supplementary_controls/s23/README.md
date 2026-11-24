# S23 Random Controls

Use Python 3.10 with NumPy 1.26.4, pandas 2.3.3 and scikit-learn 1.7.2.
From the extracted directory, run:

```bash
python generate_controls.py --input-dir inputs --summary expected_S23.csv --output new_run
```

Choose a new output directory. Input NPZ files contain coordinates, ordered spot
identifiers and computed spot-level CFAR scores. No expression preprocessing or
model fitting is performed by this script.

Outputs are `random_region_members.csv.gz`, `summary_comparison.csv` and
`result.json`. The supplied `regenerated/` directory contains 55,000 controls
across 110 section-size comparisons. The generator uses seven non-self nearest
neighbors per spot before graph symmetrization and 500 controls per comparison.
All 110 random-control medians agree with the supplied summary within 4.45e-16.

Input and source hashes are recorded in `regenerated/result.json` and
`provenance.json`, which also documents the origin of the supplied memberships.
