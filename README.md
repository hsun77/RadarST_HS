# RadarST

RadarST recovers compact tissue structures within spatial transcriptomic cluster maps and characterizes their local molecular features. It combines guarded molecular contrast, connected region proposals, topology-aware initialization and conditional refinement of the spatial partition.

## Installation

Use Python 3.10. Model files are distributed with Git LFS.

```bash
git lfs install
git clone https://github.com/hsun77/RadarST_HS.git
cd RadarST_HS
git lfs pull
python -m pip install -r requirements.txt
```

## Run the example

The included breast cancer example contains expression-derived features, spatial coordinates, an upstream GraphST partition and connected candidate regions.

```bash
python run_radarst.py \
  --input-dir example_data/map \
  --molecular-dir example_data/molecular \
  --output-root ../radarst_output \
  --dataset EXT04_GSM6592049_M2 \
  --workflow GraphST
```

Run this command from the repository root. Choose an output directory outside the repository that does not already exist. The command prints the path to the refined label table. The same run also saves selected objects, action decisions and model and input hashes.

The label table contains `spot_id`, `refined` and `edited`. Each retained object receives a distinct label; other spots retain their upstream assignments.

## Inputs

The command accepts a prepared candidate state consisting of:

- ordered spot identifiers, coordinates, spatial edges, expression principal components and one upstream partition;
- connected candidate memberships and their molecular and geometric features;
- molecular object evidence and the initialized object selection.

File formats and row-alignment requirements are described in [DATA_SCHEMA.md](DATA_SCHEMA.md). Preprocessing and initialization code is provided in `src/initialization/` and `src/frontend_helpers/`.

## Models and training

The supplied utility and event models were fitted using the 12 development sections. They predict the effects of complete map actions from 201 observable features. Conditional refinement considers additions, removals and substitutions over at most six rounds.

Training code is in `training/`. Feature order, supervised targets, section-exclusion routes and model provenance are recorded in `configs/` and `models/`. Development evaluation uses section-excluded models; external inference uses the models fitted on all development sections. Cohort training and analysis require the expression data, annotations, candidate states and fold models identified by these records.

## Repository contents

| Directory | Contents |
|---|---|
| `src/` | Molecular detection, initialization and conditional refinement |
| `models/` | Fitted models and provenance |
| `configs/` | Feature definitions, training routes and execution settings |
| `example_data/` | Prepared real-data example |
| `training/` | Supervised target preparation and model fitting |
| `results/` | Map-level metrics, component-level outcomes and supplementary tables |
| `parameter_sensitivity_30/` | Expression-weight and retention sensitivity analyses |
| `validation_completion/` | Matched controls, iteration analyses and molecular attribution |
| `supplementary_controls/s23/` | Size-matched connected-region controls |
| `tests/` | Example-output reproducibility test |

The benchmark records cover 168 development maps and 154 external maps. Cohort summaries average components within each map, workflows within each section, and sections with equal weight. Component-level tables include improved, unchanged and decreased outcomes.

## Test the example

```bash
python -m unittest discover -s tests -v
```

The test runs the supplied example and compares the resulting labels, selected objects and decisions with their recorded checksums. `SHA256SUMS` lists checksums for the distributed files.

## License

RadarST code is distributed under the [MIT License](LICENSE). Included third-party components retain their own license notices.
