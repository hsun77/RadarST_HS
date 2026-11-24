# Input data

RadarST uses aligned expression-derived features, spatial coordinates, one upstream cluster map and connected candidate regions. All input files for a map must use the same ordered spot and candidate identifiers.

| File | Contents |
|---|---|
| `MAP.npz` | Unique string spot identifiers; an N-by-2 coordinate array `xy`; integer upstream labels `base`; spatial edge pairs `edge`; expression principal-component scores `pcs` |
| `CANDIDATES.parquet` | Unique full `candidate_id`, `proposal_type`, `member_spots` and `n_spots` for each connected candidate |
| `BASE_FEATURES.parquet` | 147 observable candidate features in the same row order as `CANDIDATES.parquet` |
| `INITIAL_SELECTED.parquet` | Candidate identifiers selected during initialization |
| `GENE_OBJECT_EVIDENCE.parquet` | Molecular object evidence in the same candidate row order |

`member_spots` identifies candidate members using the spot identifiers in `MAP.npz`. Members must exist in that map, and candidate identifiers must be unique. The example map is in `example_data/map/`; its molecular evidence is in `example_data/molecular/`.

The model bundles specify the ordered 201-feature prediction vector. Dataset names, workflow names and candidate identifiers serve as alignment and routing metadata. Inference features are calculated from expression, coordinates, the upstream partition and the current object selection. Reference pathology is used to construct supervised training targets and to evaluate predictions separately.

## Model routing

The included deployment models use all 12 development sections. Reproducing development evaluation requires the corresponding section-excluded models and training-state routes in `configs/development_model_registry.json` and `configs/training_routes.json`. External maps use the development-fitted deployment models.

## Label output

The label CSV contains:

- `spot_id`: the aligned input spot identifier;
- `refined`: the predicted cluster label after conditional refinement;
- `edited`: whether the spot belongs to a selected object.

Selected-object and decision tables retain full candidate identities and membership. Model hashes and execution metadata accompany each run.
