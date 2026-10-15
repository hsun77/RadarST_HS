# Training from development inputs

Training and inference have different input contracts. The original teacher
uses **14 distinct workflow maps for each included development section**.
Ordinary inference uses a trained model and **one upstream map**. It does not
need the other maps, reference annotations or teacher files.

## Original teacher computation

The public adapter invokes the original functions in this order:

1. Generate the dense connected proposals from expression and coordinates.
2. `teacher_sensor.add_sensor_features` (57D): encode the declared maps, use
   up to eight nearest spatial neighbours outside each candidate, and calculate
   boundary, island and rare-cluster support across all declared workflows.
   The original left join on `candidate_id` is retained, including its possible
   many-to-many expansion. No historical rows or ID mappings are supplied.
3. `teacher_evidence.prepare_candidate_evidence` (70B): read only the original
   safe-column whitelist, calculate within-section/proposal-family percentiles
   and the original weighted evidence score.
4. `teacher_generation` (71B): apply the original relaxed evidence gate,
   connectivity/risk scoring, utility threshold and atomic budgeted acceptance
   separately for each upstream map.
5. `teacher_targets.load_teacher_keys` (01): apply the original safety filter
   and match accepted canonical member sets to `teacher_E7_selected`.

The actual historical teacher input was 71B acceptance; the `E7` target-column
name does not imply another teacher. The original 70A/71A computational constants
are distributed as `radarst/teacher_protocol.json`, without author paths or
historical data. The fixed support denominator is 14; this interface rejects a
single-map or arbitrary-map-count replacement rather than changing that rule.

Neither reference overlap nor the historical 57A/57D supervised predictions are
inputs to this teacher. Those prediction columns were explicitly forbidden by
the original whitelist. Their historical parent CSV is therefore not needed:
new proposals and their truth-independent sensor columns are sufficient. The
path adapter replaces only private directory rewriting and input discovery;
the extracted computational function bodies remain unchanged.

## Manifest

```yaml
samples:
  - dataset: development_section_01
    cohort: development
    expression: inputs/section_01/filtered_feature_bc_matrix.h5
    coordinates: inputs/section_01/coordinates.csv
    reference: inputs/section_01/reference_components.csv
    upstream:
      - workflow: GraphST
        labels: inputs/section_01/GraphST.csv
      - workflow: STAGATE
        labels: inputs/section_01/STAGATE.csv
      # Declare all remaining workflows explicitly, for 14 distinct maps.
```

Add each independent development section as another sample. All paths resolve
relative to the manifest. Map records preserve their declared order and may
specify `barcode_column` and `label_column` (defaults `spot_id,pred_label`).
Expression is 10x HDF5; coordinates are `spot_id,x,y`. An optional
`feature_expression` and `feature_kind: h5ad` select the existing expression-only
feature view. Every input map must align to the same unique spot IDs.

Reference columns are `spot_id,truth_label,truth_component_id`. Explicit
component IDs preserve target tie behaviour. Normalize excluded uncertainty
labels to `UNASSIGNED`; do not turn them into negatives. Exact reference sizes
1 through 10 are retained by the original target functions. Signed utility is
candidate best component IoU minus upstream best IoU for that same component;
nonnegative utility is its positive part. The teacher target is distinct.

## Complete fold-isolated training

```bash
PYTHONHASHSEED=0 radarst train-inputs --manifest training_manifest.yaml \
  --output trained_models --mode loso_and_full --audit-access
```

For PowerShell, set `$env:PYTHONHASHSEED='0'` before invoking `radarst`.
Modes are `loso`, `full`, and `loso_and_full`. To run just one fold:

```bash
PYTHONHASHSEED=0 radarst train-inputs --manifest training_manifest.yaml \
  --output one_fold --mode loso --held-out development_section_01 --audit-access
```

Each fold launches fresh preparation and fitting processes. Exclusion occurs
before teacher maps, reference files and target tables are read for that fold.
No full-fit substitute is used to evaluate a held-out section. All input samples
must be explicitly `cohort: development`; external entries are rejected. Fixed
source hyperparameters are used without threshold or parameter selection.
Files must be truthfully identified by the data provider: the software cannot
detect undisclosed donor duplicates or a falsely declared cohort.

Outputs under each `LOSO_holdout_<section>` or `full` directory include fresh
teacher evidence/acceptance, training tables, their hashes, commands, exit codes,
preparation access logs when requested, and models. The model files are under
`models/full/selector_heads.joblib` and `models/full/expected_gain.joblib`.
Despite the inner directory name, the metadata and both bundles of an outer
LOSO job explicitly identify the held-out section and only included samples.
Use those fold bundles with the `internal` inference profile. Only the outer
`full` job fits all development samples and is suitable for external inference.

Tiny/degenerate training inputs without both teacher classes, positive utility
and harm examples fail rather than fabricate a compatible estimator. The
original deterministic row sampling, group weighting, three selector heads
and expected-gain/harm ensembles are retained. Outputs are user-generated;
no model weights or real training tables are distributed.

## Separate teacher or supplied supervision

```bash
PYTHONHASHSEED=0 radarst generate-teacher --manifest training_manifest.yaml \
  --held-out development_section_01 --output teachers --audit-access
```

This stage never opens reference annotations. For existing legally supplied
teacher files, add `teacher_acceptance: inputs/teacher.csv` per sample, then:

```bash
PYTHONHASHSEED=0 radarst prepare-training --manifest training_manifest.yaml \
  --teacher-source provided --held-out development_section_01 --output prepared
radarst train --config training.yaml
```

Teacher CSV fields are `dataset_id,method,proposal_type,member_spots,
fragmentation_risk,erosion_risk,boundary_leakage_risk,evidence_score`.
Each file belongs only to its declared sample. This route trusts the user's
teacher provenance; it cannot certify an externally supplied file's isolation.
To generate teachers during preparation instead, use `--teacher-source generate`.

A fitting configuration is:

```yaml
manifest: prepared/training_tables.yaml
output: fold_models
mode: full
held_out_dataset: development_section_01
```

The excluded-dataset entry is checked against the prepared manifest. For already
prepared multi-sample tables, the older `train` modes `loso`/`loso_and_full`
exclude held-out tables before reading and fitting; that alone does **not** prove
that an external preparation stage excluded that sample. Prefer `train-inputs`
for verifiable raw-input fold isolation.

## Reproducibility boundary

This exposes the actual computation, not an archive of previous experimental
states. The original short ID depends on the Python hash state; this release
requires the documented startup seed and does not restore an unknown historical
secret. Newly fitted weights are not claimed to equal any previous serialization.
This training interface does not certify historical sample isolation, nor does
it automatically reproduce previously archived paper outputs. Legal raw inputs,
their workflow order, reference provenance and compatible dependencies remain
the user's responsibility.
