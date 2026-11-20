"""MILP vs DP on real cohorts: same cohorts, same models, same utility floor.

For every cohort and model this solves the LDA-exp problem three ways and
measures each allocation's disparity on the true needs:

  DP        allocations_parallel._solve_dp on np.rint(predicted need)
  MILP-int  allocations_milp.solve_milp on the same rounded needs
            (must reach the DP's disparity and utility)
  MILP-raw  solve_milp on the unrounded predicted needs, which the DP cannot take

Cohorts are drawn exactly as run_variability.py / allocations_parallel.py draw
them (uniform group split, default_rng(0), group.sample(random_state=iteration)),
so the first cohorts of each seed coincide with the earlier runs.

    uv run python scripts/compare_milp_dp.py --size 250 --k 25 --iters-per-seed 20 --n-top 5
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

from run_perturbation_sweep import ALLOCATION_EPS, GROUP_COL, MODEL_CLASS, NEED_COL  # noqa: E402
from run_variability import load_top_models  # noqa: E402
from src.allocations_parallel import _solve_dp  # noqa: E402
from src.allocations_milp import solve_milp  # noqa: E402


def sample_cohorts(frame, n, iters_per_seed):
    """The uniform-split sampler of allocations_parallel.get_allocations."""
    rng = np.random.default_rng(0)
    cohorts = []
    for seed in sorted(frame["seed"].unique()):
        split = frame[frame["seed"] == seed]
        population = split[split["person_id"] >= 0]
        groups = [population[population[GROUP_COL] == g] for g in (0, 1)]
        for iteration in range(iters_per_seed):
            count_0 = int(rng.choice(range(1, n)))
            counts = [count_0, n - count_0]
            cohort = pd.concat([
                group.sample(n=count, random_state=iteration)
                for count, group in zip(counts, groups)
            ]).sample(frac=1, random_state=iteration, ignore_index=True)
            cohorts.append((int(seed), iteration, cohort))
    return cohorts


def true_disparity(selected, group, need):
    sel = np.zeros(len(group), bool)
    sel[selected] = True
    g0, g1 = sel & (group == 0), sel & (group == 1)
    if not g0.any() or not g1.any():
        return np.nan, int(g0.sum()), int(g1.sum())
    return abs(need[g1].mean() - need[g0].mean()), int(g0.sum()), int(g1.sum())


def solve_cohort(seed, iteration, cohort, k, n_top, time_limit):
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
        common = {"seed": seed, "iteration": iteration, "model": j, "k": k,
                  "cohort_n0": len(groups[0]), "cohort_n1": len(groups[1]),
                  "rmse_rounded": float(np.sqrt(np.mean((rounded - need) ** 2))),
                  "rmse_raw": float(np.sqrt(np.mean((raw - need) ** 2)))}

        def record(name, selected, seconds, given, extra=None):
            d, k0, k1 = true_disparity(selected, group, need)
            d_given, _, _ = true_disparity(selected, group, given)
            rows.append({**common, "algorithm": name, "selected": sorted(int(i) for i in selected),
                         "absolute_disparity": d, "disparity_on_given": d_given,
                         "utility": float(p[selected].sum()), "k_0": k0, "k_1": k1,
                         "seconds": seconds, **(extra or {})})

        record("OPT", top_k, 0.0, rounded)
        t0 = time.perf_counter()
        dp_sel, fallback = _solve_dp(groups, p.tolist(), [int(v) for v in rounded], k, floor, top_k)
        record("DP", dp_sel, time.perf_counter() - t0, rounded, {"fallback": fallback})
        sel, info = solve_milp(groups, p, rounded, k, floor, time_limit=time_limit)
        record("MILP-int", sel if sel is not None else top_k, info["seconds"], rounded,
               {"fallback": sel is None, "splits": info["splits"], "status": info["status"]})
        sel, info = solve_milp(groups, p, raw, k, floor, time_limit=time_limit)
        record("MILP-raw", sel if sel is not None else top_k, info["seconds"], raw,
               {"fallback": sel is None, "splits": info["splits"], "status": info["status"]})
    return rows


def summarise(df):
    piv = df.pivot_table(index=["seed", "iteration", "model"], columns="algorithm",
                         values=["absolute_disparity", "disparity_on_given", "utility", "seconds"])
    same_disp = np.isclose(piv["disparity_on_given"]["DP"], piv["disparity_on_given"]["MILP-int"], atol=1e-7)
    same_util = np.isclose(piv["utility"]["DP"], piv["utility"]["MILP-int"], atol=1e-7)
    sets = df[df.algorithm.isin(["DP", "MILP-int"])].pivot_table(
        index=["seed", "iteration", "model"], columns="algorithm", values="selected", aggfunc="first")
    same_set = (sets["DP"].map(tuple) == sets["MILP-int"].map(tuple))
    lines = [
        f"cohort-model pairs: {len(piv)}",
        f"MILP-int reaches the DP's disparity on the given needs: {same_disp.mean():.1%}",
        f"MILP-int reaches the DP's utility:                      {same_util.mean():.1%}",
        f"MILP-int selects the identical set of people:           {same_set.mean():.1%}",
        "",
        "mean disparity on TRUE needs:",
    ]
    for alg in ("OPT", "DP", "MILP-int", "MILP-raw"):
        d = piv["absolute_disparity"][alg]
        lines.append(f"  {alg:9s} {d.mean():.3f}  (sd {d.std():.3f})")
    better = piv["absolute_disparity"]["MILP-raw"] - piv["absolute_disparity"]["DP"]
    lines += [
        "",
        f"MILP-raw minus DP on true needs: mean {better.mean():+.3f}, "
        f"raw better in {(better < -1e-9).mean():.0%}, worse in {(better > 1e-9).mean():.0%}, equal in {(abs(better) <= 1e-9).mean():.0%}",
        "",
        "seconds per solve (one cohort, one model):",
    ]
    for alg in ("DP", "MILP-int", "MILP-raw"):
        s = piv["seconds"][alg]
        lines.append(f"  {alg:9s} mean {s.mean():.3f}  median {s.median():.3f}  max {s.max():.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--iters-per-seed", type=int, default=20)
    ap.add_argument("--n-top", type=int, default=5, help="lowest-validation-loss models per split")
    ap.add_argument("--n-models", type=int, default=1000)
    ap.add_argument("--need-threshold", type=int, default=1)
    ap.add_argument("--pred-method", default="average")
    ap.add_argument("--time-limit", type=float, default=0.5,
                    help="seconds per split MILP solve; proving optimality can take hours")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default="data/results/milp")
    args = ap.parse_args()

    start = time.time()
    fp = ROOT / ("data/predictions/"
                 f"test_bootstrap_{MODEL_CLASS}_{args.pred_method}_qualified_gagne_{args.need_threshold}.csv")
    frame, chosen = load_top_models(fp, args.n_models, args.n_top, round_needs=False)
    race = pd.read_csv(ROOT / "data/obermeyer/obermeyer_data_cleaned.csv", usecols=["person_id", GROUP_COL])
    frame = frame.merge(race, on="person_id", how="left")
    frame[GROUP_COL] = frame[GROUP_COL].fillna(0).astype(int)
    cohorts = sample_cohorts(frame, args.size, args.iters_per_seed)
    print(f"[start] n={args.size} k={args.k} cohorts={len(cohorts)} models={args.n_top} "
          f"loaded in {(time.time()-start)/60:.1f} min", flush=True)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(solve_cohort)(seed, it, cohort, args.k, args.n_top, args.time_limit)
            for seed, it, cohort in cohorts)
    df = pd.DataFrame([row for rows in results for row in rows])
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"size-{args.size}_k-{args.k}_iters-{args.iters_per_seed}_models-{args.n_top}"
    df.to_csv(out_dir / f"compare_{tag}.csv", index=False)
    text = summarise(df)
    (out_dir / f"summary_{tag}.txt").write_text(text + "\n")
    print(text)
    print(f"[done] {len(df):,} rows in {(time.time()-start)/60:.1f} min -> {out_dir / f'compare_{tag}.csv'}", flush=True)


if __name__ == "__main__":
    main()
