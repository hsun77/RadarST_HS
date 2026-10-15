# RadarST: Post-clustering refinement for spatial transcriptomics

## Overview

RadarST refines one existing spatial clustering map using guarded local molecular
contrast, connected multiscale proposals, fitted candidate ranking and
topology-aware integration. It complements upstream clustering; it does not run
or replace the upstream clustering method. Reference annotations are used only
in supervised training or separate evaluation, never in ordinary inference.

This repository contains source code, configuration templates and synthetic
tests. It does not distribute expression data, pretrained weights or experiment
outputs. A compatible pair of trained model bundles is required for inference.

## Installation

Use Python 3.10 in a new environment. Replace `<repository-url>` with the actual
repository URL when available.

```bash
git clone <repository-url> RadarST
cd RadarST
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e ".[test]"
radarst --help
```

For the tested direct dependency versions, use `requirements-lock.txt`. A
conda environment definition is also provided in `environment.yml`. No GPU is
required. NumPy/BLAS uses 20 threads and estimator inference uses one OpenMP
thread, matching the implemented execution protocol.

## Requirements

Key dependencies are NumPy, pandas, SciPy, scikit-learn, Scanpy, AnnData, h5py,
PyArrow, joblib and threadpoolctl. Models serialized with another scikit-learn
version are not assumed compatible. Only load trusted joblib files: loading a
pickle-based model can execute code.

## Input Format

- **Expression:** a 10x Genomics filtered feature-barcode HDF5 matrix. The detector
  reads counts and applies its own existing normalization and variable-gene
  selection. An optional separate H5AD expression view can supply feature PCA;
  only its matrix and index names are read, not pathology metadata.
- **Coordinates:** CSV with unique `spot_id,x,y` columns, in the desired spot
  order. Coordinates must be finite and use one consistent spatial frame.
- **Upstream labels:** CSV with unique `spot_id,pred_label` columns covering
  the input spots. Other column names can be declared explicitly.
- **Models:** two user-supplied joblib dictionaries: the expected-gain ensemble
  and the selector teacher/signed-utility heads. See [models](docs/models.md).

Missing expression spots, duplicate IDs and incompatible model schemas produce
errors. The legacy positional upstream-alignment fallback is available only by
explicit opt-in; it is never silently enabled for a new user's map.

## Quick Start

```bash
radarst synthetic --output synthetic_inputs
PYTHONHASHSEED=0 radarst prepare-features --expression synthetic_inputs/filtered_feature_bc_matrix.h5 \
  --coordinates synthetic_inputs/coordinates.csv \
  --upstream-labels synthetic_inputs/upstream_labels.csv \
  --dataset synthetic --workflow example --config configs/inference.yaml \
  --output synthetic_features
```

This creates explicitly synthetic inputs and exercises the real detector,
proposal generator and feature construction. It does **not** pretend to run
model-based refinement without weights.

## Inference

```bash
PYTHONHASHSEED=0 radarst infer \
  --expression inputs/filtered_feature_bc_matrix.h5 \
  --coordinates inputs/coordinates.csv \
  --upstream-labels inputs/upstream_labels.csv \
  --ranker-model user_models/expected_gain.joblib \
  --selector-model user_models/selector_heads.joblib \
  --dataset section_01 --workflow GraphST \
  --config configs/inference.yaml --output new_results
```

In PowerShell, set `$env:PYTHONHASHSEED='0'` before starting `radarst`. The
profile in the configuration selects an existing input/graph branch; it is
not an outcome-based model-selection option. Use `internal` for a held-out
development model, `external_first` for the Visium branch, or `external_second`
for the existing array-coordinate branch. Supply the appropriate already
trained models. Dataset/workflow identifiers participate in the original
deterministic selector seed and output naming, not model feature identity.

Inference writes fresh CFAR scores, candidate memberships, features, scores,
selector inputs, selected regions, refined labels and run metadata beneath the
specified new output directory. It does not search for models or cached
features elsewhere. Nonempty output directories are rejected.

## Training

The original teacher classifier, signed-utility heads and expected-gain/harm
ensembles are included, with their existing targets, sampling and hyperparameters.

```bash
PYTHONHASHSEED=0 radarst train-inputs --manifest training_manifest.yaml \
  --output trained_models --mode loso_and_full --audit-access
```

Training requires development expression, coordinates, reference components and
**14 distinct upstream workflow maps per training section**. The original
multi-map teacher is generated from these maps, not from reference overlap.
Reference components then define the original utility targets. For each LOSO
fold, the complete held-out section is excluded before teacher or target
construction, as well as fitting. External data are not training inputs.
Ordinary model-based inference still needs only **one** upstream map.

The optional `generate-teacher` command performs just the truth-independent
multi-map stage. The existing `prepare-training` / `train` interface continues
to support explicitly supplied original-format teacher decisions. See
[training](docs/training.md) for schemas, commands and reproducibility boundaries.

An entirely artificial end-to-end training example is available:

```bash
python examples/synthetic_training_demo.py --output artificial_training
PYTHONHASHSEED=0 radarst train-inputs --manifest artificial_training/manifest.yaml \
  --output artificial_models --mode loso_and_full --audit-access
```

These generated inputs test software execution only; they are not biological
validation and are not used to select algorithm settings.

## Configuration

Use `configs/inference.yaml` and `configs/training.yaml` as templates. Paths in
configuration files resolve relative to the configuration file; no author-specific
directories are required. Inference validates the implemented fixed parameter
values rather than silently accepting a different algorithm. The original
candidate-ID rule depends on process hash state; set the process environment
before Python starts. No historical ID mapping or candidate cache is supplied.

## Evaluation

```bash
radarst evaluate --coordinates inputs/coordinates.csv \
  --reference inputs/reference.csv --predictions new_results/refined_labels.csv \
  --output evaluation_results
```

Reference CSV columns are `spot_id,truth_label`. Evaluation computes connected
component best-IoU, coverage, matched-region purity, ARI, NMI and AMI. Explicit
positive-label filtering and graph radius are evaluation options; they do not
change inference. See [evaluation](docs/evaluation.md).

## Citation

Software citation metadata are provided in `CITATION.cff`. No article DOI or
publication status is asserted. Add the associated article's verified citation
when it becomes available.

## License

The project code is released under the MIT License with author authorization.
Dependencies retain their own licenses; see `THIRD_PARTY_NOTICES.md`. Models,
datasets and external annotations are not included or licensed by this repository.
