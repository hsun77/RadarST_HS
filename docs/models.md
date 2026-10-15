# Model inputs

Model weights are not distributed. Supply trusted joblib files explicitly.
No default search path, download, random fallback or historical-state loader is
used. The code validates model schemas and feature order before prediction.

The expected-gain dictionary contains `positive_models`, `rank_models`,
`harm_models`, `proposal_types` and the ordered 104 `feature_columns`.
The selector-head dictionary contains `teacher`, `signed_utility_classifier`,
`signed_utility_regressor`, `proposal_types`, `harm_weight=0.5` and
`extra_utility_budget=0.01`. Each selector head uses 28 ordered features.

For profile `internal`, both dictionaries must have `held_out_dataset` equal
to the supplied dataset ID and `training_datasets` excluding that ID. A full-fit
bundle is not an acceptable substitute for a held-out model. External inference
uses models trained on the development collection only.

`radarst train` writes one compatible pair under `full/` or each
`LOSO_holdout_<dataset>/`. Its metadata record training dataset identifiers,
input checksums and model checksums. These are user-generated outputs and are
not committed to this code-only repository. Cross-version pickle compatibility
is not guaranteed; use the pinned software environment.
