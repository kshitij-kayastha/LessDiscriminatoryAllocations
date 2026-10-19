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

## Seeding

`--seed` defaults to `0`, which seeds NumPy before cohort sampling and makes a
run reproducible. Pass `--seed=-1` for the notebook's behaviour, where each run
draws different cohorts.

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
