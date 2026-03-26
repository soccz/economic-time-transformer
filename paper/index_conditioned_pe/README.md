# Index-Conditioned PE Project

This folder is a fresh paper track that separates the new main idea from the earlier
FF3 residual momentum drafts.

The working thesis is:

> broad-market index flow can be treated as a conditioning signal for temporal
> coordinates, and that conditioned coordinate system changes how a hybrid
> Transformer-TCN model routes global context, local motifs, and predictive
> uncertainty.

Current honest status:

- the paper path is stock-only and standalone
- explicit stock-index conditioning is useful relative to `static`
- `concat_a` is currently the strongest ranking baseline on repeated-seed checks
- a true coordinate-warp variant (`flow_pe`) is now implemented, but it has not
  beaten `concat_a` on ranking and its state-swap response is still weak
- the probabilistic `CVAE` path runs, but H3 is not yet calibrated well enough to headline

Files:

- `00_topic_decision.md`: candidate topic comparison and the final main-topic lock
- `01_hypotheses.md`: final research question and aligned hypotheses
- `02_experiment_matrix.md`: ablations, diagnostics, metrics, and go/no-go rules
- `03_code_alignment.md`: what can be reused from the current codebase and what must change
- `05_actual_hybrid_results.md`: current paper-facing neural results and locked claim
- `06_stock_only_boundary.md`: hard boundary for paper-facing code, data, and language
- `07_coordinate_warp_results.md`: follow-up results for the true coordinate-warp `flow_pe` variant
- `08_routing_mask_results.md`: quantitative check showing current attention masking is not yet a valid explanation diagnostic
- `99_worklog.md`: running record for decisions and next actions

Ground rule:

- only one main question is allowed
- probabilistic decoding is subordinate to the main representation claim
- FF3 residual / Fama-MacBeth material is retained only as an application path, not the paper center
- paper-facing code must remain stock-only and self-contained
