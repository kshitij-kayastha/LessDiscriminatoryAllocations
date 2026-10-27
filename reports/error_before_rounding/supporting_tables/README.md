# Supporting tables

These aggregate tables support the accompanying report. Each allocation observation is an eligible model/sample pair. Comparisons use the same seed, sample iteration, and model at each grid spacing.

- `disparity_by_beta_delta.csv`: true-need disparity and utility summaries for all 15 settings.
- `disparity_references.csv`: reference summaries; OPT disparity excludes undefined observations.
- `reference_sources.csv`: provenance of each reference allocation family.
- `paired_delta_comparisons.csv`: finer-minus-coarser disparity changes; negative is improvement. A changed allocation means at least one selected person differs.
- `paired_seed_comparisons.csv`: paired disparity changes averaged within each seed.
- `error_summary.csv`: MAE and RMSE diagnostics averaged over model/sample pairs. Each RMSE is first calculated within a 100-person sample, then averaged; it is not a single pooled RMSE.
- `rounding_error_details.csv`: error erasure and amplification fractions over individual/model/sample observations. For β > 0, all original adjusted errors in this run are nonzero, so `fraction_erased` is also the fraction of nonzero errors erased. For β = 0 it is zero by definition. `population_mae` uses the entire sample; `selected_mae` uses selected people only. Comparisons use a numerical tolerance of 1e-12.

Standard deviations use ddof = 1 and are descriptive, not confidence intervals. The five seeds and repeated models/samples induce dependence between observations.
