"""Is the variability of the DP's disparity driven by n, by k, or by the number
of cohorts? LDA-exp on the needs predicted by the models themselves (rounded
expected need, as the DP uses), no perturbation, for a grid of cohort sizes n
and fixed resource counts k. OPT and LDA-tru run on the same cohorts as
references.

Each split's N lowest-validation-loss models supply their own probabilities
and predicted needs. Every setting is run once with --iters-per-seed cohorts
per seed; the smaller cohort counts are the first 10, 25, 50 of them, which
summarise_variability.py reads off the same file.

    uv run python scripts/run_variability.py --sizes 250 500 1000 --k 25 50 100 \
        --iters-per-seed 100 --n-top 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_perturbation_sweep import (  # noqa: E402
    ALLOCATION_EPS, GROUP_COL, MODEL_CLASS, NEED_COL, SweepAllocations, metric_rows)


def load_top_models(fp, n_models, n_top):
    """Columns p_j / m_j (probabilities) and exp_need_m_j (rounded predicted
    need) for each split's n_top lowest-validation-loss models, j = 1..n_top."""
    models = [f"m_{i}" for i in range(1, n_models + 1)]
    need_cols = [f"exp_need_{m}" for m in models]
    df = pd.read_csv(fp, usecols=models + need_cols + ["y", "person_id", NEED_COL, "seed"],
                     dtype={c: "float32" for c in models + need_cols})
    parts, chosen = [], {}
    for seed, split in df.groupby("seed"):
        losses = split.loc[split["person_id"] == -1, models].iloc[0].astype(float).sort_values()
        top = list(losses.index[:n_top])
        chosen[int(seed)] = [(m, float(losses[m])) for m in top]
        part = split[["y", "person_id", NEED_COL, "seed"]].copy()
        cols = {}
        for j, m in enumerate(top, 1):
            cols[f"p_{j}"] = split[m].astype("float32")
            cols[f"m_{j}"] = split[m].astype("float32")
            # the DP sees rounded needs; metric_rows measures RMSE on this column
            cols[f"exp_need_m_{j}"] = np.rint(split[f"exp_need_{m}"].astype(np.float64))
        parts.append(pd.concat([part, pd.DataFrame(cols, index=part.index)], axis=1))
    return pd.concat(parts).sort_index(), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--k", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--iters-per-seed", type=int, default=100)
    ap.add_argument("--n-top", type=int, default=100,
                    help="lowest-validation-loss models per split")
    ap.add_argument("--n-models", type=int, default=1000, help="models in the prediction file")
    ap.add_argument("--group-split", default="uniform", choices=["uniform", "population"])
    ap.add_argument("--need-threshold", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--pred-method", default="average", choices=["average", "argmax"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default="data/results/variability")
    args = ap.parse_args()

    start = time.time()
    fp = ROOT / ("data/predictions/"
                 f"test_bootstrap_{MODEL_CLASS}_{args.pred_method}_qualified_gagne_{args.need_threshold}.csv")
    frame, chosen = load_top_models(fp, args.n_models, args.n_top)
    race = pd.read_csv(ROOT / "data/obermeyer/obermeyer_data_cleaned.csv", usecols=["person_id", GROUP_COL])
    frame = frame.merge(race, on="person_id", how="left")
    frame[GROUP_COL] = frame[GROUP_COL].fillna(0).astype(int)
    print(f"[start] sizes={args.sizes} k={args.k} iters={args.iters_per_seed} "
          f"models={args.n_top} split={args.group_split}  loaded in {(time.time()-start)/60:.1f} min",
          flush=True)
    for seed, ms in chosen.items():
        pop = frame[(frame["seed"] == seed) & (frame["person_id"] >= 0)]
        err = pop[f"exp_need_m_1"] - pop[NEED_COL]
        print(f"  seed {seed}: best {ms[0][0]} (val loss {ms[0][1]:.4f}), model {args.n_top} {ms[-1][1]:.4f}; "
              f"group sizes {pop[GROUP_COL].value_counts().sort_index().to_list()}; "
              f"population RMSE of best model {np.sqrt((err**2).mean()):.3f}", flush=True)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # cheapest settings first, so results arrive early
    settings = sorted(((n, k) for n in args.sizes for k in args.k), key=lambda s: s[0] * s[1] ** 2)
    for n, k in settings:
        t0 = time.time()
        rate = (k + 0.5) / n   # the allocator floors rate * n, so this gives exactly k
        a = SweepAllocations(
            df=frame, sample_size=n, selection_rate=rate, qualification_col="y",
            need_col=NEED_COL, group_col=GROUP_COL, rashomon_eps=1.0,
            allocation_eps=ALLOCATION_EPS, n_models=args.n_top,
            iters_per_seed=args.iters_per_seed, n_jobs=args.n_jobs,
            show_progress=False, group_split=args.group_split)
        if a.k != k:
            sys.exit(f"allocator k={a.k} for n={n}, k={k}")
        opt, tru, exp, people = a.get_allocations(exp_need_col="exp_need_{model}")
        common = {"sample_size": n, "selection_rate": k / n, "iters_per_seed": args.iters_per_seed,
                  "group_split": args.group_split, "n_top": args.n_top,
                  "need_threshold": args.need_threshold, "pred_method": args.pred_method}
        rows = []
        for name, table in (("OPT", opt), ("LDA-tru", tru), ("LDA-exp", exp)):
            rows += metric_rows(frame, people, table, name, k, common)
        df = pd.DataFrame(rows)
        tag = (f"size-{n}_k-{k}_iters-{args.iters_per_seed}_models-{args.n_top}"
               f"_split-{args.group_split}_{args.pred_method}_gagne_{args.need_threshold}")
        df.to_csv(out_dir / f"var_{tag}.csv", index=False)
        per_cohort = (df.groupby(["algorithm", "seed", "iteration"])["absolute_disparity"]
                        .mean().groupby("algorithm"))
        print(f"  n={n} k={k}: {len(people)} cohorts, {len(df):,} rows, {(time.time()-t0)/60:.1f} min; "
              + ", ".join(f"{alg} {m:.3f} ± {s:.3f}" for alg, m, s in
                          zip(per_cohort.mean().index, per_cohort.mean(), per_cohort.std())),
              flush=True)

    print(f"[done] in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
