# Expression-weight and retention sensitivity

The sensitivity analysis evaluates 30 combinations of initialization settings across 12 development sections and 14 upstream workflows, yielding 5,040 complete map predictions.

Expression weights are 0.70, 0.75, 0.80, 0.85 and 0.90. Accepted-area retention fractions are 0.95, 0.96, 0.97, 0.98, 0.99 and 1.00. The principal setting uses expression weight 0.80 and retention 0.97.

For expression weight w, the candidate ranking score combines w times the molecular percentile and (1-w) times the expected-gain rank, with a cross-component risk penalty of 0.025. Expected-gain coefficients are rounded to the prescribed decimal values. Initialization uses 96 assignment attempts, connected candidates, spatial nonoverlap and the composition template. Retention packs complete objects within the specified fraction of assigned member area.

Each configuration uses the same candidate pool, model feature order, section-excluded models, deterministic ordering and six-round conditional refinement. The prediction records contain all 30 label vectors, ordered spot identifiers, selected candidate indices, actions and source hashes. At the principal setting, candidate membership and labels are checked against the corresponding benchmark predictions.

Reference labels are read for evaluation after prediction. Evaluation uses the source-specific reference masks and connected-component graphs. Components are averaged within a workflow and section, workflows within a section, and the 12 sections equally. Exact-size summaries include sections with an eligible component at that size. Minimum workflow gain is the minimum of the 14 section-equal workflow means.

`sensitivity.py` generates predictions and `evaluate_sensitivity.py` calculates map-level and component-level results. `test_evaluator.py` compares the sparse evaluator with explicit set-based IoU calculations. Source data and model paths are specified in these scripts and the input identity records.

`evaluation/` contains the complete paired results, including negative and zero changes. `predictions/` contains the label vectors and selection records. Supplementary Table S31 summarizes the parameter grid. The external cohort is evaluated at the principal configuration; it is not part of this parameter grid.
