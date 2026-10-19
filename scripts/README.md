# Experiment scripts

Command-line equivalents of the notebooks in `test/`, so a full sweep can run
unattended instead of editing a config cell and keeping a kernel alive.

## Setup

```bash
uv venv --python 3.14 && uv sync
```

## Inputs

`data/` is gitignored, so these have to be in place first:

| File | Produced by |
|---|---|
| `data/obermeyer/obermeyer_data_cleaned.csv` | `test/00-data_cleaning.ipynb` |
| `data/predictions/test_bootstrap_nn_qualified_gagne_{1,2,3}.csv` | `scripts/make_predictions.py` |

```bash
uv run python scripts/make_predictions.py --all --model-class nn
```

Takes about 3.5 h with the three targets in parallel. `gagne_2` and `gagne_3`
are slower than `gagne_1`: the rarer positive class needs more iterations to
converge. Output is deterministic given the same scikit-learn version — these
files were verified identical to an independently generated set.

## Running

One configuration:

```bash
uv run python scripts/run_experiment.py \
    --selection-rate 0.5 --qualification-column qualified_gagne_1
```

All nine, one process each:

```bash
bash scripts/run_all.sh
tail -f logs/run_*.log
```

Defaults match the agreed settings (`TEST_SIZE=250`, `NUM_MODELS=100`,
`RASHOMON_EPSILON=0.01`, `ALLOCATION_DELTA=0.1`, `ITERATIONS_PER_SPLIT=10`), and
output paths are identical to the notebook's, so results are interchangeable.

`--skip-existing` skips any configuration whose comparison CSV already exists,
which makes an interrupted sweep resumable.

On macOS, wrap the sweep in `caffeinate` so idle sleep cannot stall it:

```bash
nohup caffeinate -is bash scripts/run_all.sh > logs/run_all.log 2>&1 &
```

## Seeding, and reusing cohorts

`--seed` defaults to `0`, which seeds NumPy before cohort sampling and makes a
run reproducible. Pass `--seed=-1` for the notebook's behaviour, where each run
draws different cohorts.

Because the notebook never seeded the draw, two runs of the same configuration
see different people and cannot be compared cohort by cohort. To put two runs on
identical cohorts, point the second at the first run's `people_*.csv`:

```bash
uv run python scripts/run_experiment.py \
    --selection-rate 0.25 --qualification-column qualified_gagne_1 \
    --people-file data/cohorts/people_..._rate-0.25_size-250.csv \
    --cohort-label kshitij
```

`--cohort-label` is added to the output filenames, so these results sit beside
the originals rather than overwriting them. The run fails loudly if a cohort is
missing, is the wrong size, or names a person absent from that split.

## Metrics from saved allocations

`scripts/compute_metrics.py` derives further metrics from
`data/results/allocations/` without re-running anything — a sweep takes about ten
seconds across all nine configurations.

```bash
uv run python scripts/compute_metrics.py --self-test   # check the maths
uv run python scripts/compute_metrics.py               # write diversity_*.csv
```

Allocations are expanded per model first: a row in the allocation files is one
*distinct* selection, and its `models` column says how many models produced it,
so a selection found by 40 models counts 40 times. That makes `p` a per-person
selection probability and matches "two random allocations from the set".

| Metric | Meaning |
|---|---|
| `pairwise_difference` | Mean symmetric difference between two allocations, scaled to [0, 1]. 0 = identical, 1 = disjoint |
| `pairwise_consistency` | Share of people given the same outcome by two random allocations. The paper's homogenization measure |
| `heterozygosity` | Mean of 2p(1−p) over people |
| `selection_entropy` | Mean binary Shannon entropy of p, in bits |
| `share_chance_ge_10pct` | Fraction of the cohort selected by at least 10% of models |
| `systemic_exclusion_all` | Fraction of the cohort selected by no model |
| `systemic_exclusion_qualified` | Same, among qualified people only — the quantity the paper reports |

The last two differ: `metrics.py:57` divides by everyone, while the paper's
10–30% figure is among the qualified. Both are kept so they are not confused.

**Uniqueness is not diversity.** `num_unique_allocations` counts selections that
differ at all, so two allocations differing in one person count as two. Every
measure above instead asks *how far apart* they are, and the two disagree: the DP
produces fewer unique allocations than top-k in 8 of 9 configurations, but scores
better on every diversity measure in 9 of 9.

## Measured runtimes

100 cohorts × 100 models per configuration, at `TEST_SIZE=250`:

| selection rate | k | one config, alone | nine in parallel |
|---|---|---|---|
| 0.10 | 25 | — | ~3.8 h |
| 0.25 | 62 | — | ~12–13 h |
| 0.50 | 125 | 17.7 h | ~23–24 h |

About 120 core-hours for the full sweep, ~24 h wall clock. Nine concurrent
processes run roughly 1.3–1.6× slower each than one alone, since the DP's
dictionaries of tuples make it memory-bandwidth bound. Each process is
single-threaded and peaks around 0.25 GB.

Selection cost per model per cohort, DP against top-k:

| selection rate | k | top-k | DP |
|---|---|---|---|
| 0.10 | 25 | 0.69 ms | 0.98 s |
| 0.25 | 62 | 0.53 ms | 3.36 s |
| 0.50 | 125 | 0.53 ms | 5.95 s |

Holding the selection *rate* fixed, cost grows roughly with the cube of n,
because k = rate × n grows with n.

## Two observations from running the sweep

**`RASHOMON_EPSILON` excludes almost nothing.** Across the 100 bootstrap models
the validation-loss spread is 0.0059–0.0086 for `qualified_gagne_1`,
0.0069–0.0107 for `_2` and 0.0077–0.0141 for `_3` — about the size of ε = 0.01
itself. The filter keeps 100 of 100 models for the first target and 98–100 for
the other two, so what the sweep compares is effectively the sampling method
rather than the Rashomon construction.

**Cohort group composition is drawn uniformly at random.**
`Allocations.get_allocations` uses `k_0 = np.random.choice(range(1, TEST_SIZE))`,
so a 250-person cohort can contain anywhere from 1 to 249 patients of either
group, every split equally likely, against about 11% in the population. Where a
cohort holds very few patients of one group, that group's mean need is an average
over a handful of people and the per-cohort disparity is dominated by noise. The
natural-proportion alternative is on the line commented out directly above.
