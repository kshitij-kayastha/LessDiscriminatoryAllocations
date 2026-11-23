"""What does a per-split time limit cost the MILP?

For every cohort and model this solves the LDA-exp problem with the MILP once
exactly (no time limit) and once per time limit in --limits, on the unrounded
predicted needs (--needs raw, the MILP's reason to exist) or on np.rint of them
(--needs int). It records, per limit: whether the allocation is the exact one,
the disparity on the given needs (what the solver optimises), the disparity on
the true needs (what we pay), the utility and the seconds. The DP and OPT are
recorded alongside for reference.

    uv run python scripts/milp_time_limit.py --size 250 --k 25 --iters-per-seed 10 --n-top 2 \
        --limits 0.05 0.1 0.5 --solver gurobi
"""

import argparse
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_milp_dp import sample_cohorts, true_disparity  # noqa: E402
from run_perturbation_sweep import ALLOCATION_EPS, GROUP_COL, MODEL_CLASS, NEED_COL  # noqa: E402
from run_variability import load_top_models  # noqa: E402
from src.allocations_parallel import _solve_dp  # noqa: E402
from src.allocations_milp import solve_milp  # noqa: E402


def solve_cohort(seed, iteration, cohort, k, n_top, limits, solver, which):
    group = cohort[GROUP_COL].to_numpy()
    need = cohort[NEED_COL].to_numpy(dtype=np.float64)
    groups = [np.flatnonzero(group == g).tolist() for g in (0, 1)]
    rows = []
    for j in range(1, n_top + 1):
        p = cohort[f"p_{j}"].to_numpy(dtype=np.float64)
        series = pd.Series(p)
        top_k = series.nlargest(k).index.to_list()
        floor = series.loc[top_k].sum().item() - k * ALLOCATION_EPS
        raw = cohort[f"exp_need_m_{j}"].to_numpy(dtype=np.float64)
        rounded = np.rint(raw)
        given = raw if which == "raw" else rounded
        common = {"seed": seed, "iteration": iteration, "model": j, "k": k, "needs": which,
                  "cohort_n0": len(groups[0]), "cohort_n1": len(groups[1])}

        def record(name, limit, selected, seconds, extra=None):
            d, k0, k1 = true_disparity(selected, group, need)
            d_given, _, _ = true_disparity(selected, group, given)
            rows.append({**common, "algorithm": name, "time_limit": limit,
                         "selected": sorted(int(i) for i in selected),
                         "absolute_disparity": d, "disparity_on_given": d_given,
                         "utility": float(p[selected].sum()), "k_0": k0, "k_1": k1,
                         "seconds": seconds, **(extra or {})})

        record("OPT", np.nan, top_k, 0.0)
        t0 = time.perf_counter()
        dp_sel, _ = _solve_dp(groups, p.tolist(), [int(v) for v in rounded], k, floor, top_k)
        record("DP", np.nan, dp_sel, time.perf_counter() - t0)
        exact, info = solve_milp(groups, p, given, k, floor, solver=solver)
        exact = exact if exact is not None else top_k
        record("MILP", np.nan, exact, info["seconds"], {"status": info["status"]})
        for limit in limits:
            sel, info = solve_milp(groups, p, given, k, floor, solver=solver, time_limit=limit)
            sel = sel if sel is not None else top_k
            record("MILP", limit, sel, info["seconds"],
                   {"status": info["status"], "same_as_exact": sel == exact})
    return rows


def summarise(df):
    key = ["seed", "iteration", "model"]
    exact = df[(df.algorithm == "MILP") & df.time_limit.isna()].set_index(key)
    dp = df[df.algorithm == "DP"].set_index(key)
    opt = df[df.algorithm == "OPT"].set_index(key)
    lines = [f"cohort-model pairs: {len(exact)}   needs given to the MILP: {df['needs'].iloc[0]}",
             f"OPT        true disparity {opt.absolute_disparity.mean():.3f}",
             f"DP         true disparity {dp.absolute_disparity.mean():.3f}  utility {dp.utility.mean():.3f}  {dp.seconds.mean():.3f} s",
             f"MILP exact true disparity {exact.absolute_disparity.mean():.3f}  utility {exact.utility.mean():.3f}  "
             f"given {exact.disparity_on_given.mean():.4f}  {exact.seconds.mean():.2f} s (max {exact.seconds.max():.1f})",
             "", "per time limit (seconds per split solve):"]
    for limit in sorted(df.time_limit.dropna().unique()):
        lim = df[(df.algorithm == "MILP") & (df.time_limit == limit)].set_index(key).loc[exact.index]
        d_given = lim.disparity_on_given - exact.disparity_on_given
        d_true = lim.absolute_disparity - exact.absolute_disparity
        lines.append(
            f"  limit {limit:5.2f}s: same allocation as exact {lim.same_as_exact.mean():5.1%}; "
            f"hit the limit {(lim.status == 1).mean():5.1%}; "
            f"given disparity exact {exact.disparity_on_given.mean():.4f} -> {lim.disparity_on_given.mean():.4f} "
            f"(worse in {(d_given > 1e-7).mean():.0%}, max +{d_given.max():.4f}); "
            f"true disparity {exact.absolute_disparity.mean():.3f} -> {lim.absolute_disparity.mean():.3f} "
            f"(mean diff {d_true.mean():+.3f}, worse in {(d_true > 1e-9).mean():.0%}, better in {(d_true < -1e-9).mean():.0%}); "
            f"utility {exact.utility.mean():.3f} -> {lim.utility.mean():.3f}; "
            f"{lim.seconds.mean():.2f} s per cohort-model")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--iters-per-seed", type=int, default=10)
    ap.add_argument("--n-top", type=int, default=2)
    ap.add_argument("--n-models", type=int, default=1000)
    ap.add_argument("--need-threshold", type=int, default=1)
    ap.add_argument("--pred-method", default="average")
    ap.add_argument("--limits", type=float, nargs="+", default=[0.05, 0.1, 0.5])
    ap.add_argument("--solver", default="gurobi", choices=["highs", "gurobi"])
    ap.add_argument("--needs", default="raw", choices=["raw", "int"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default="data/results/milp")
    args = ap.parse_args()

    start = time.time()
    fp = ROOT / ("data/predictions/"
                 f"test_bootstrap_{MODEL_CLASS}_{args.pred_method}_qualified_gagne_{args.need_threshold}.csv")
    frame, _ = load_top_models(fp, args.n_models, args.n_top, round_needs=False)
    race = pd.read_csv(ROOT / "data/obermeyer/obermeyer_data_cleaned.csv", usecols=["person_id", GROUP_COL])
    frame = frame.merge(race, on="person_id", how="left")
    frame[GROUP_COL] = frame[GROUP_COL].fillna(0).astype(int)
    cohorts = sample_cohorts(frame, args.size, args.iters_per_seed)
    print(f"[start] n={args.size} k={args.k} cohorts={len(cohorts)} models={args.n_top} solver={args.solver} "
          f"needs={args.needs} limits={args.limits}", flush=True)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(solve_cohort)(seed, it, cohort, args.k, args.n_top, args.limits, args.solver, args.needs)
            for seed, it, cohort in cohorts)
    df = pd.DataFrame([row for rows in results for row in rows])
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"size-{args.size}_k-{args.k}_iters-{args.iters_per_seed}_models-{args.n_top}_{args.solver}_{args.needs}"
    df.to_csv(out_dir / f"timelimit_{tag}.csv", index=False)
    text = summarise(df)
    (out_dir / f"timelimit_summary_{tag}.txt").write_text(text + "\n")
    print(text)
    print(f"[done] {len(df):,} rows in {(time.time()-start)/60:.1f} min -> {out_dir / f'timelimit_{tag}.csv'}", flush=True)


if __name__ == "__main__":
    main()
