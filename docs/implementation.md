# Implementation and operational constraints

The `_formal` modules contain project algorithm functions extracted without
function-body changes. Imports are package-local dependency namespaces. The
frontend adapter changes input/model path resolution and exposes its pre-model
feature-construction prefix; it does not change arithmetic, candidate ordering,
model scoring, selector attempts or atomic retention.

Detector expression normalization and feature PCA are distinct existing views.
The detector retains the guarded neighbour-rank reference and robust PC scoring.
The sparse connected proposal ladder, all singleton and edge proposals, original
candidate-ID functions, member canonicalization, feature insertion order,
28-feature selector heads and 104-feature expected-gain ensemble are retained.
No graph-shell or adaptive-growth development variant is substituted.

The process hash environment must be selected before Python starts. Short IDs
are not assumed globally unique; the implementation retains its existing
canonical membership and row identity fields. No changed ID function or archived
ID mapping is introduced by the package. The public raw-input path does not
manufacture duplicate rows from a stored collision-state table. These operational
constraints are not a guarantee of a previously serialized experiment state.

After scoring, model-generated acceptance records supply the existing selector
composition. The original exact-token selector uses 96 attempts and its existing
identifier-derived seed. Retention is applied after full assignment as an atomic
spot budget: floor(0.97 times selected spot count, with the implemented numerical
tolerance). Whole candidates are retained in the original score/key order.

Optional `--audit-access` enforces a read allowlist for scientific inputs and
restricts writes to the new output directory. It records Python file operations
and explicit HDF5 dataset reads. It is not a kernel-level system-call sandbox.
The H5AD reader accesses only expression and index names during inference.

The multi-map teacher is a training-only stage. Its original 57D sensor
aggregation and short-ID join, 70B safe-column evidence and 71B risk-aware
acceptance are now exposed by `generate-teacher` and `train-inputs`. The latter
excludes each held-out section before generating any teacher or utility targets
for its fold. This does not change the one-map inference pipeline or establish
identity with a historical hash state. See [training](training.md).

Small sample sizes, duplicate coordinates, invalid IDs or degenerate fitted
models can make the formal numerical procedures inapplicable. Such cases should
be reported as errors, not handled by a silently different detector or model.
