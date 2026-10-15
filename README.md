# RadarST

RadarST refines spatial transcriptomics clustering maps to improve the representation of compact tissue components.
It combines guarded local molecular contrast, connected multiscale proposals and topology-aware integration.
It operates downstream of an existing clustering workflow and does not rerun upstream clustering.

## Installation

Use Python 3.10. In an activated virtual environment:

```bash
git clone https://github.com/hsun77/RadarST_HS.git
cd RadarST_HS
python -m pip install -e ".[test]"
radarst --help
```

Conda users can use `conda env create -f environment.yml`, then `conda activate radarst`.
Dependencies are specified in [pyproject.toml](pyproject.toml); no GPU is required.

## Input Format

- Expression: a 10x Genomics filtered feature-barcode HDF5 count matrix.
- Coordinates: CSV with unique `spot_id,x,y` columns.
- Upstream map: CSV with unique `spot_id,pred_label` columns covering the input spots.
- Inference models: compatible, trusted expected-gain and selector model bundles supplied by the user; weights are not bundled.

See [model requirements](docs/models.md) and [input and configuration details](docs/implementation.md).

## Quick Start / Inference

Set `PYTHONHASHSEED=0` before starting the CLI (`$env:PYTHONHASHSEED='0'` in PowerShell).
Inference requires one upstream map and compatible models, but no reference annotations.

```bash
PYTHONHASHSEED=0 radarst infer \
  --expression inputs/filtered_feature_bc_matrix.h5 \
  --coordinates inputs/coordinates.csv --upstream-labels inputs/upstream_labels.csv \
  --ranker-model models/expected_gain.joblib --selector-model models/selector_heads.joblib \
  --dataset section_01 --workflow GraphST \
  --config configs/inference.yaml --output new_results
```

Use a new output directory; select the input profile in [configs/inference.yaml](configs/inference.yaml) and follow the [model requirements](docs/models.md).

## Training

Training requires Development expression, coordinates, reference annotations and 14 distinct upstream workflow maps per section.
LOSO excludes the held-out section before teacher generation, target construction and model fitting; External data are not training inputs.

```bash
PYTHONHASHSEED=0 radarst train-inputs --manifest training_manifest.yaml \
  --output trained_models --mode loso_and_full --audit-access
```

See [training](docs/training.md) for manifest schemas, teacher generation and model outputs.

## Documentation

[Implementation and configuration](docs/implementation.md) | [Models](docs/models.md) | [Training](docs/training.md) | [Evaluation](docs/evaluation.md).
Synthetic examples are provided in [examples/](examples/); run unit tests with `python -m pytest`.

## Citation

Software citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

[MIT License](LICENSE). Dependencies retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
