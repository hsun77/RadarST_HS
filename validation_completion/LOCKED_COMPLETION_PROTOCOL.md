# RadarST manuscript experiment completion

Locked before new reference-based control evaluation, 2026-11-05. Algorithm,
models, 168 Development maps and 154 External maps are immutable. This is
completion of fixed scientific controls, not algorithm or threshold selection.

## Original-paper inventory and scope

Already completed for actual final RadarST: all 322 full predictions; component
and sample/workflow paired endpoints; exact/0.8 and size stratification;
ARI/NMI/AMI, harm, dilution, purity, actual footprint; section bootstrap;
initialization ablation; External leave-one-section-out description; executable
prepared-state example and reference-free inference audit. Upstream-only DLPFC
and annotation census can be reused without claiming that RadarST was run there.

Missing and undertaken here: final-map matched randomization, matched molecular
ordering, census and CFAR attribution of CURRENT selected physical objects,
current membership/shape verification of historical biological cases, and
scope-correct sensitivity/robustness accounting.

## Fixed-selection controls

All 322 registered pairs, no selective omission. Seed 20260812, pair seed =
SHA256(dataset|workflow|seed), first eight bytes little endian modulo 2**32-1.
1000 random complete selections per map; exactly match final count, proposal
type-by-size quotas and actual candidate-member area. Require frozen candidate
identity, 1–10 size, spatial connectivity and nonoverlap. Generate without
reference access; globally hash all memberships before any new label evaluation.
Reconstruct complete upstream-plus-selected-object labels, and evaluate on the
original source-specific reference masks and prediction graph. Independent
evaluator parity required. Aggregate components within map, workflows within
section, then sections equally. Paired replicate number defines a cohort null
replicate. Report empirical upper-tail probability (1+exceedances)/(1000+1),
descriptive 2.5–97.5% null interval, not a bootstrap confidence interval.

Final conditional objects have no channel identity. Channel matching is NOT
APPLICABLE, never fabricated. Static legal-pool randomization is not a
randomization of learned state-dependent eligibility. The matched molecular
control uses original per-spot molecular percentile, descending order, same
quotas and nonoverlap, 100 deterministic seeded ordering restarts. Unsolvable
quotas are INFEASIBLE; no relaxation. Neither control chooses a new algorithm.

## Biological checks

Deduplicate CURRENT selected objects by section and exact sorted spot members;
retain workflow recurrence, proposal type and source IDs. Audit every spot,
selected shape and final written connected instance. Reuse persisted original
spot-level expression/PCA/CFAR signed gene arrays only after spot alignment,
source provenance and singleton CFAR numerical parity. Re-aggregate region
medians for every CURRENT shape including those absent from the old 20561 panel.
Old panel statistics are not called current population statistics.

Use frozen local MSigDB 2026.1 Hallmark/Reactome/GOBP, sizes 15–500, gene
universe observed in at least 12 of 23 sections. Same fixed 1000 prerank
permutations and 10000 section-block sign flips; all genes within section share
the flip. Attribution fidelity gate retains original thresholds. Current
size-matched connected CFAR controls: 500 regions per section/size using the
original seven-nonself-neighbor symmetric graph, seed 20260805. Labels enter
only descriptive context/evaluation, never model decisions. This is molecular
interpretation of detector contrast, not independent biology or causal disease
validation. No new case or pathway selected by reconstruction IoU.

## Sensitivity boundary

The old 5 expression-weight × 6 retention grid describes initialization, not
the current conditional policy. It cannot be copied or presented as final-model
robustness. First inspect exact interfaces and compute cost. Any executed grid
must keep all conditional heads, routes and decision rules fixed, run complete
writeback and report every grid point without selecting a winner. If not
faithfully executable, report the concrete gap and retain an explicit pending
status; never fill with historical values or invented zeros.

## Protection and resources

New files exclusively in this directory; G project, frozen research sources,
models and manuscript sources read-only. At most eight total compute workers,
one BLAS thread per worker, no GPU. Persist logs, failures, inputs and output
hashes. No new training, public data acquisition, External tuning or figure
redrawing. Unknown donor identity remains UNKNOWN. All studies remain exposed
Development / historically examined cross-cohort descriptive evidence.
