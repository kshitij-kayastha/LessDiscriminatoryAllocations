"""Generate allocations for one trained pool and summarise them.

Runs src/allocations_parallel.Allocations over the predictions written by
scripts/train_models.py, saves the four allocation tables, and builds the
comparison table the notebooks stop short of.

Two things to know about the defaults:

  * n_jobs defaults to 4 in the allocator, which uses a quarter of this
    machine. This script passes -1 unless told otherwise.
  * the DP optimises np.rint(expected need), so the disparity measured on
    predicted needs is computed on the rounded values here. Measuring the
    raw expected_need column instead reports a quantity nothing optimises,
    which is what made argmax look four times better than it is.

    uv run python scripts/run_allocations.py --need-threshold 1 \
        --pred-method average --selection-rate 0.1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocations_parallel import Allocations  # noqa: E402

MODEL_CLASS = "nn_multiclass"
NEED_COL = "gagne_sum_t"
EXP_NEED_COL = "exp_need_{model}"
GROUP_COL = "race"
RASHOMON_EPS = 0.01
ALLOCATION_EPS = 0.1
ALGORITHMS = (("OPT", "opt"), ("LDA-tru", "lda_tru_need"), ("LDA-exp", "lda_exp_need"))


def load_predictions(fp, n_models):
    """Read only the columns needed, at float32, so a 3 GB file stays manageable."""
    header = pd.read_csv(fp, nrows=0).columns
    models = [f"m_{i}" for i in range(1, n_models + 1)]
    exp_needs = [EXP_NEED_COL.format(model=m) for m in models]
    missing = [c for c in models + exp_needs if c not in header]
    if missing:
        sys.exit(f"{fp.name}: missing {len(missing)} columns, first few {missing[:5]}")
    keep = models + exp_needs + ["y", "person_id", NEED_COL, "seed"]
    dtypes = {c: "float32" for c in models + exp_needs}
    return pd.read_csv(fp, usecols=keep, dtype=dtypes), models, exp_needs


def comparison_rows(df, people, allocations, cohort_cols, k, args, tag):
    models, exp_needs = cohort_cols
    model_pos = {m: j for j, m in enumerate(models)}
    rows = []
    for _, info in people.iterrows():
        seed, iteration = info["seed"], info["iteration"]
        split = df.loc[(df["seed"] == seed) & (df["person_id"] >= 0)].set_index("person_id")
        cohort = split.loc[info["people"]]
        pos = {pid: i for i, pid in enumerate(cohort.index)}

        probabilities = cohort[models].to_numpy()
        # the quantity the DP actually balances
        rounded_needs = np.rint(cohort[exp_needs].to_numpy(dtype=np.float64))
        need = cohort[NEED_COL].to_numpy(dtype=np.float64)
        group = cohort[GROUP_COL].to_numpy()
        qualified = cohort["y"].to_numpy(dtype=np.float64)

        for algorithm, _ in ALGORITHMS:
            table = allocations[algorithm]
            if table.empty:
                continue
            cohort_allocations = table.loc[(table["seed"] == seed)
                                           & (table["iteration"] == iteration)]
            if cohort_allocations.empty:
                continue
            num_unique = len(cohort_allocations)
            num_models = sum(len(ms) for ms in cohort_allocations["models"])

            for _, allocation in cohort_allocations.iterrows():
                rowsel = [pos[p] for p in allocation["selected"]]
                cols = [model_pos[m] for m in allocation["models"]]
                g0 = group[rowsel] == 0
                g1 = group[rowsel] == 1

                mean_0, mean_1 = need[rowsel][g0].mean(), need[rowsel][g1].mean()
                ratio = np.nan
                if np.isfinite([mean_0, mean_1]).all() and mean_0 and mean_1:
                    ratio = max(mean_1 / mean_0, mean_0 / mean_1)

                block = rounded_needs[np.ix_(rowsel, cols)]
                exp_disparity = np.abs(block[g1].mean(axis=0) - block[g0].mean(axis=0))
                utility = probabilities[np.ix_(rowsel, cols)].mean(axis=0)
                realized = qualified[rowsel].mean()

                for offset, model in enumerate(allocation["models"]):
                    rows.append({
                        "method": tag,
                        "algorithm": algorithm,
                        "qualification_column": f"qualified_gagne_{args.need_threshold}",
                        "pred_method": args.pred_method,
                        "seed": seed,
                        "iteration": iteration,
                        "model": model,
                        "k": k,
                        "n_models": args.n_models,
                        "epsilon": ALLOCATION_EPS,
                        "expected_utility": utility[offset],
                        "realized_utility": realized,
                        "absolute_disparity": abs(mean_1 - mean_0),
                        "disparity_ratio": ratio,
                        "absolute_disparity_expected": exp_disparity[offset],
                        "num_rashomon_models": num_models,
                        "num_unique_allocations": num_unique,
                        "unique_allocation_ratio": num_unique / num_models if num_models else np.nan,
                    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-threshold", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--pred-method", required=True, choices=["average", "argmax"])
    ap.add_argument("--selection-rate", type=float, required=True, choices=[0.1, 0.25, 0.5])
    ap.add_argument("--method", default="bootstrap")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--iters-per-seed", type=int, default=10)
    ap.add_argument("--n-models", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--predictions", default=None,
                    help="override the input path; used to check the script at a tiny size")
    args = ap.parse_args()

    qualification_col = f"qualified_gagne_{args.need_threshold}"
    stem = f"test_{args.method}_{MODEL_CLASS}_{args.pred_method}_{qualification_col}"
    tag = (f"{stem}_rate-{args.selection_rate}_size-{args.sample_size}"
           f"_iters-{args.iters_per_seed}_models-{args.n_models}{args.suffix}")

    fp_predictions = (Path(args.predictions) if args.predictions
                      else ROOT / f"data/predictions/{stem}.csv")
    fp_input = ROOT / "data/obermeyer/obermeyer_data_cleaned.csv"
    for fp in (fp_predictions, fp_input):
        if not fp.exists():
            sys.exit(f"missing input: {fp}")

    alloc_dir = ROOT / "data/results/allocations"
    alloc_dir.mkdir(parents=True, exist_ok=True)
    fp_comparison = ROOT / f"data/results/comparison_{tag}.csv"
    fp_comparison.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print(f"[start] {tag}  n_jobs={args.n_jobs}", flush=True)

    df, models, exp_needs = load_predictions(fp_predictions, args.n_models)
    race = pd.read_csv(fp_input, usecols=["person_id", GROUP_COL])
    df = df.merge(race, on="person_id", how="left", validate="many_to_one")
    # the sentinel rows carry no person, so their group label is filled in
    df[GROUP_COL] = df[GROUP_COL].fillna(0).astype(int)
    print(f"  loaded {df.shape[0]:,} rows x {df.shape[1]:,} columns "
          f"in {(time.time()-start)/60:.1f} min", flush=True)

    a = Allocations(
        df=df, sample_size=args.sample_size, selection_rate=args.selection_rate,
        qualification_col="y", need_col=NEED_COL, group_col=GROUP_COL,
        rashomon_eps=RASHOMON_EPS, allocation_eps=ALLOCATION_EPS,
        n_models=args.n_models, iters_per_seed=args.iters_per_seed,
        n_jobs=args.n_jobs, show_progress=True,
    )
    opt, tru, exp, people = a.get_allocations(exp_need_col=EXP_NEED_COL)
    print(f"  {len(people)} cohorts, k = {a.k}, "
          f"allocations done at {(time.time()-start)/60:.1f} min", flush=True)

    tables = dict(zip([name for name, _ in ALGORITHMS], (opt, tru, exp)))
    for (name, slug), table in zip(ALGORITHMS, (opt, tru, exp)):
        table.to_csv(alloc_dir / f"{slug}_{tag}.csv")
    people.to_csv(alloc_dir / f"people_{tag}.csv")
    pd.DataFrame(a.allocation_diagnostics).to_csv(
        ROOT / f"data/results/diagnostics_{tag}.csv", index=False)

    results = comparison_rows(df, people, tables, (models, exp_needs), a.k, args, stem)
    results.to_csv(fp_comparison, index=False)

    print(f"[done] {tag}  {len(results):,} rows -> {fp_comparison.name}  "
          f"({(time.time()-start)/60:.1f} min)", flush=True)
    summary = results.groupby("algorithm")[
        ["absolute_disparity", "absolute_disparity_expected", "disparity_ratio",
         "realized_utility", "num_unique_allocations"]].mean().round(4)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
