# Controls and molecular analyses

This directory contains candidate-selection controls, iteration analyses and molecular characterization for the 168 development and 154 external maps.

## Candidate-selection controls

`complete_controls.py` constructs 1,000 connected, nonoverlapping selections per map. Each selection matches RadarST in candidate count, proposal type by exact size and unique member-spot area. Selections are written back to the complete upstream partition before evaluation. Matched replicate indices define the 1,000 cohort-level realizations. A molecular-ordering control uses the same candidate pool and matching constraints.

Results are provided in `MATCHED_CONTROL_ALL322_RESULTS.csv`, `MATCHED_CONTROL_COHORT_SUMMARY.csv` and `MOLECULAR_ORDER_PAIRED_BOOTSTRAP.csv`. The tables retain each reported metric and both cohorts.

## Iteration analysis

`iteration_ablation.py` reconstructs the initialized partition and each of six conditional-refinement rounds. Results include complete map-level, section-level and cohort-level summaries in the `ITERATION_ABLATION_*.csv` tables.

## Molecular characterization

Molecular analyses deduplicate selected objects by section and exact spot membership. Gene contributions use the signed detector deviations and corresponding principal-component loadings. Pathway analyses use the specified MSigDB libraries, 1,000 preranked permutations and 10,000 section-block sign flips. Connected random-region comparisons use 500 regions per section and size.

The accompanying tables contain gene and pathway contributions, context summaries, random-region comparisons and the correspondence of illustrated regions to selected objects. Both the 20,561-region illustration panel and the 23,961 selected physical objects are identified in their respective analyses.

## Running the analyses

Run the commands from this directory after making the registered cohort inputs available at the paths specified by the scripts:

```bash
python complete_controls.py generate --workers 8
python complete_controls.py evaluate --workers 8
python complete_biology.py
python iteration_ablation.py generate --workers 4
python iteration_ablation.py evaluate --workers 4
python summarize_completion.py
```

These cohort analyses require the source expression data, reference registry, candidate pools, predicted maps and spot-level detector arrays recorded in `INPUT_PROVENANCE.json`. The small example under `example_data/` supplies one prepared map for inference.

Cohort statistics use section-equal aggregation. Bootstrap resampling retains repeated workflows within each section; random-selection intervals describe the control distribution separately. All paired outcomes are retained in the result tables.
