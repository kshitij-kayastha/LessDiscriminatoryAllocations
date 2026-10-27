# Controlled prediction errors before rounding

**Experiment report · 25 October 2026**

## Main finding

Finer discretization produced only a small, inconsistent reduction in LDA-exp’s absolute disparity. At full prediction error (β = 1), changing the need grid from δ = 1 to δ = 0.2 reduced mean disparity from **1.650 to 1.631**, approximately **1.2%**. Both remain well above LDA-tru’s **0.241**. Rounding does hide many small individual prediction errors, but these results suggest it is not the main explanation for the disparity gap in this setting.

## Question and design

We tested whether rounding expected needs to integers hides or exaggerates the effect of prediction errors on allocations. For individual *i*, let hᵢ denote the true need and μᵢ the model’s **unrounded expected need**, computed from its multiclass probability distribution. We first adjust the prediction error:

$$z_i(\beta)=h_i+\beta(\mu_i-h_i).$$

Thus, β = 0 uses true needs, β = 1 retains the full prediction error, and intermediate values retain that fraction of each individual’s original error. This is a controlled diagnostic using known true needs; it is not a method for correcting unknown needs at deployment.

We then discretize once, using a grid spacing δ:

$$H_i=\operatorname{rint}\!\left(\frac{z_i(\beta)}{\delta}\right),\qquad \widetilde h_i=\delta H_i.$$

The DP receives the integer grid index Hᵢ. Scaling all needs by the same positive factor preserves their disparity ordering, so no second rounding to whole need units is necessary. `rint` rounds to the nearest integer, with exact halfway cases going to the even integer. Smaller δ represents adjusted needs more finely, but still involves rounding.

| Setting | Value |
|---|---|
| Sample and allocation sizes | n = 100; k = 10 (10% selection) |
| Prediction model | Bootstrap multiclass neural networks; probability-weighted expected needs |
| Error retained, β | 0, 0.25, 0.50, 0.75, 1 |
| Grid spacing, δ | 1, 0.5, 0.2 |
| Repetitions | 5 seeds × 5 samples per seed = 25 sample instances |
| Models | Up to 100 candidates per seed, filtered by the saved Rashomon criterion |
| Eligible observations | 2,490 model/sample pairs per β/δ setting |
| Other parameters | Rashomon ε = 0.01; allocation ε = 0.1; α replacement disabled |
| Need and group variables | `gagne_sum_t` and `race` |

The same models and samples are paired across all 15 settings. Every allocation is evaluated using **true needs**: absolute disparity is the absolute difference between the two groups’ mean true needs among selected individuals. The DP’s utility constraint is held fixed across settings.

## Results

![Absolute disparity versus beta for delta 1, 0.5, and 0.2, with standard deviation bands and LDA-exp, LDA-tru, and OPT references.](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/reports/error_before_rounding/absolute_disparity_vs_beta.png)

*Figure 1. Mean absolute disparity against the fraction of prediction error retained before rounding. Shading shows ±1 standard deviation, with lower limits clipped at zero. LDA-exp is the β = 1, δ = 1 endpoint; LDA-tru is the β = 0 endpoint. OPT is the saved top-k baseline. The nearly overlapping curves indicate that the grid choice has a much smaller effect on average disparity than the amount of prediction error retained.*

| β | δ = 1 | δ = 0.5 | δ = 0.2 |
|---:|---:|---:|---:|
| 0 | 0.241 ± 0.724 | 0.241 ± 0.724 | 0.241 ± 0.724 |
| 0.25 | 0.491 ± 0.699 | 0.503 ± 0.685 | 0.502 ± 0.688 |
| 0.50 | 0.788 ± 0.754 | 0.787 ± 0.765 | 0.777 ± 0.764 |
| 0.75 | 1.178 ± 0.897 | 1.135 ± 0.874 | 1.128 ± 0.876 |
| 1 | 1.650 ± 1.152 | 1.628 ± 1.129 | 1.631 ± 1.138 |

Values are mean ± sample standard deviation across eligible model/sample pairs. Reference disparities are **LDA-exp: 1.650 ± 1.152**, **LDA-tru: 0.241 ± 0.724**, and **OPT: 1.969 ± 1.411**.

### Does finer discretization reduce disparity?

Only modestly on average, and not uniformly. Comparing δ = 0.2 with δ = 1, the mean change is **+0.011** at β = 0.25, **−0.010** at β = 0.50, **−0.049** at β = 0.75, and **−0.020** at β = 1. A negative change indicates improvement. The finest grid is also slightly worse than δ = 0.5 at β = 1.

At β = 1, the finer grid improves disparity in 45.4% of paired allocations, worsens it in 40.2%, and leaves it unchanged in 14.4%. Mean disparity improves in only two of the five seeds. Nevertheless, the selected set changes in **92.9%** of paired allocations—meaning at least one selected individual changes, not that 92.9% of individuals change. Grid resolution can therefore affect *who* is selected substantially without substantially improving average disparity.

### Is rounding hiding or exaggerating prediction errors?

It can do both. Because true needs lie on these grids, sufficiently small errors round back to the true need. Larger errors can instead be rounded farther away. The table below illustrates this at β = 0.25:

| Representation of adjusted needs | Mean absolute error | Mean within-sample RMSE | Nonzero errors erased by rounding |
|---|---:|---:|---:|
| Before rounding | 0.174 | 0.301 | — |
| δ = 1 | 0.094 | 0.315 | 91.1% |
| δ = 0.5 | 0.135 | 0.309 | 78.0% |
| δ = 0.2 | 0.153 | 0.299 | 59.9% |

Whole-integer rounding hides most small errors here: it lowers MAE by about 46%, while slightly increasing RMSE. Finer discretization represents the adjusted predictions more faithfully, but that does not translate into a comparable reduction in allocation disparity. Even δ = 0.2 still erases many small errors.

Across all three grids, disparity increases strongly as β increases. Therefore, the qualitative relationship between retained prediction error and disparity persists when rounding is made finer. These results do **not** support describing whole-integer rounding as consistently exaggerating or consistently hiding the effect on *disparity*: its direction varies with β and with the allocation.

## Interpretation and limits

The evidence supports focusing the next experiment on how prediction errors influence the allocation decision, rather than treating finer discretization alone as the main remedy. This experiment does not identify which prediction errors matter most, establish that the DP algorithm is incorrect, or guarantee that the same pattern holds at other sample sizes or selection rates.

The bands describe variation across allocations, **not confidence intervals**. Model/sample observations share models and people, so the 2,490 rows should not be treated as independent repetitions. There are only five seeds, and no statistical significance claim is made.

All DP settings have defined disparity for all 2,490 observations. OPT has defined disparity for only **1,849**; the remaining **641** select members of only one group, making the two-group mean difference undefined. They are excluded from OPT’s disparity summary, not assigned zero. Consequently, the OPT reference uses a different valid subset and is descriptive rather than a fully paired comparison.

## Saved artifacts and reproducibility

This report uses the complete 15-setting run in `size-100_models-100_iters-5_7807aca5c013`. The separate n = 250 configuration has no completed runs and is not included.

- [Experiment notebook](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/test/02-experiments_error_before_rounding.ipynb)
- [Results notebook](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/test/03-results_error_before_rounding.ipynb)
- [Experiment configuration](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/data/results/allocations/error_before_rounding/size-100_models-100_iters-5_7807aca5c013/config.json)
- [Full-precision disparity table](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/data/results/allocations/error_before_rounding/size-100_models-100_iters-5_7807aca5c013/analysis/disparity_by_beta_delta.csv)
- [Reference results](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/data/results/allocations/error_before_rounding/size-100_models-100_iters-5_7807aca5c013/analysis/disparity_references.csv)
- [Interactive version of Figure 1](/Users/kk985/PhD/Research/LessDiscriminatoryAllocations/reports/error_before_rounding/absolute_disparity_vs_beta.html)

The run-file hashes and input fingerprints were verified before preparing the report. Diagnostic and paired-comparison tables are included alongside this report in `supporting_tables/`.
