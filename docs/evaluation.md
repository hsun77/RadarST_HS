# Independent evaluation

`radarst evaluate` reads user predictions and references after inference. It
does not load any fitted model or write back into a prediction directory.

Coordinates require `spot_id,x,y`; predictions require `spot_id,pred_label`;
references require `spot_id,truth_label`. All IDs are explicitly aligned.
Disconnected pieces of one label are evaluated as separate spatial components.
The spatial graph uses median positive nearest-neighbour distance multiplied
by `--graph-scale` (default 1.25). A separate coordinate CSV and multiplier may
be specified for the reference graph when an assay uses a different frame.

For each reference component, the original implementation finds the connected
prediction with maximum IoU. Coverage is overlap divided by reference size;
matched-region purity is overlap divided by matched predicted-region size.
Macro-domain agreement uses the original ARI, NMI and AMI functions and their
unassigned-label exclusions. Degenerate macro metrics are reported as null.

Use repeated `--positive-label` flags to restrict which reference labels are
component endpoints. This does not change the macro metrics or inference.
Outputs are `component_metrics.csv` and `metrics.json`.

The original sample-equal hierarchy and section-bootstrap functions are also
available in `radarst._formal.bootstrap` for explicitly constructed event tables.
Do not treat component events or workflow repeats as independent biological
samples. No experiment endpoint tables are distributed with this repository.
