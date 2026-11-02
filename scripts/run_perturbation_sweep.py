"""DP sensitivity to random need error, on real Obermeyer cohorts.

Perturb the TRUE needs: with probability q a person's need moves by a uniform
integer step in 1..r with a random sign, clipped to 0..17. Each perturbation
draw is handed to the allocator as one "model" (D draws = D model columns), so
its parallelism, cohort sampler and bookkeeping are reused unchanged, and the
within-cohort / between-cohort split of the sd comes out as it does for the
Rashomon runs. Qualification probabilities are held fixed at each split's
best-validation-loss model, so the only thing that varies is the needs.
With --probs-models M the M lowest-validation-loss models of each split each
supply their own probabilities, and every model gets the same D perturbation
draws, so the sd across models isolates the effect of the probabilities.

OPT and LDA-tru do not depend on (q, r); they are run once per selection rate
with the same fixed probabilities (one allocation per model) and written as
reference rows.

    uv run python scripts/run_perturbation_sweep.py --sample-size 250 \
        --selection-rate 0.05 0.1 0.2 0.3 0.4 0.5 --iters-per-seed 100 --n-draws 25
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocations_parallel import Allocations, _ALGORITHMS, _ALLOCATION_COLUMNS  # noqa: E402

MODEL_CLASS = "nn_multiclass"
NEED_COL, GROUP_COL, NEED_MAX = "gagne_sum_t", "race", 17
ALLOCATION_EPS = 0.1
DEFAULT_Q = (0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_R = (1, 2, 3)
SEED_BASE = 20261031


class SweepAllocations(Allocations):
    """Allocations with a choice of cohort group split.

    uniform:    count_0 ~ U{1..n-1}, the sampler in allocations_parallel.py.
    population: count_0 ~ Binomial(n, share of group 0 in the split), clipped to
                1..n-1, i.e. i.i.d. draws from the population. Needed above
                n ~ 1100, where the uniform split asks for more group-1 people
                than a test split holds.
    """

    def __init__(self, *args, group_split="uniform", **kwargs):
        super().__init__(*args, **kwargs)
        if group_split not in ("uniform", "population"):
            raise ValueError("group_split must be 'uniform' or 'population'")
        self.group_split = group_split

    def get_allocations(self, exp_need_col=None, *, algorithms=None):
        if self.group_split == "uniform":
            return super().get_allocations(exp_need_col, algorithms=algorithms)
        # The parent's loop with the one line that draws count_0 changed.
        if algorithms is None:
            algorithms = ["OPT", "LDA-tru"] + (["LDA-exp"] if exp_need_col is not None else [])
        if isinstance(algorithms, str):
            algorithms = [algorithms]
        algorithms = tuple(name for name in _ALGORITHMS if name in set(algorithms))
        self.allocation_diagnostics = []
        allocations = {algorithm: [] for algorithm in _ALGORITHMS}
        people = []
        rng = np.random.default_rng(0)
        with self._executor() as executor:
            for seed in sorted(self.df["seed"].unique()):
                split = self.df[self.df["seed"] == seed]
                population = split[split["person_id"] >= 0]
                perf = split[split["person_id"] < 0]
                groups = [population[population[self.group_col] == g] for g in (0, 1)]
                share_0 = len(groups[0]) / len(population)
                for iteration in range(self.iters_per_seed):
                    count_0 = int(np.clip(rng.binomial(self.sample_size, share_0),
                                          1, self.sample_size - 1))
                    counts = [count_0, self.sample_size - count_0]
                    cohort = pd.concat([
                        group.sample(n=count, random_state=iteration)
                        for count, group in zip(counts, groups)
                    ]).sample(frac=1, random_state=iteration, ignore_index=True)
                    people.append({"seed": seed, "iteration": iteration,
                                   "people": cohort["person_id"].to_list()})
                    records = self._evaluate_cohort(
                        seed, iteration, cohort, perf, algorithms, exp_need_col, executor)
                    for algorithm in algorithms:
                        allocations[algorithm].extend(records[algorithm])
        return tuple(pd.DataFrame(allocations[name], columns=_ALLOCATION_COLUMNS)
                     for name in _ALGORITHMS) + (
                         pd.DataFrame(people, columns=["seed", "iteration", "people"]),)


def load_reference(fp, n_models, n_probs=1):
    """Each split's n_probs lowest-validation-loss models supply the probabilities,
    as columns p_1 (the best) .. p_{n_probs}; p_ref is an alias of p_1."""
    models = [f"m_{i}" for i in range(1, n_models + 1)]
    df = pd.read_csv(fp, usecols=models + ["y", "person_id", NEED_COL, "seed"],
                     dtype={m: "float32" for m in models})
    probs = {j: pd.Series(np.nan, index=df.index, dtype="float32")
             for j in range(1, n_probs + 1)}
    chosen = {}
    for seed, split in df.groupby("seed"):
        losses = split.loc[split["person_id"] == -1, models].iloc[0].astype(float).sort_values()
        chosen[int(seed)] = [(m, float(losses[m])) for m in losses.index[:n_probs]]
        for j, (m, _) in enumerate(chosen[int(seed)], 1):
            probs[j].loc[split.index] = split[m]
    out = df[["y", "person_id", NEED_COL, "seed"]].copy()
    for j in range(1, n_probs + 1):
        out[f"p_{j}"] = probs[j]
    out["p_ref"] = out["p_1"]
    return out, chosen


def perturb(needs, q, r, rng):
    """Move each need with probability q by a step in 1..r, random sign, clipped."""
    n = len(needs)
    move = rng.random(n) < q
    step = rng.integers(1, r + 1, n) * rng.choice([-1, 1], n)
    return np.clip(needs + np.where(move, step, 0), 0, NEED_MAX)


def model_frame(base, n_draws, q=None, r=None, n_probs=1):
    """base plus one model column per (probability model j, draw d), numbered
    m_{(j-1)*D + d}: the probabilities of p_j and, for a sweep point,
    exp_need_m_... holding perturbation draw d. Draw d is the same vector for
    every j, so the models differ only in their probabilities."""
    pop = (base["person_id"] >= 0).to_numpy()
    needs = base.loc[pop, NEED_COL].to_numpy(dtype=np.int64)
    draws = {}
    if q is not None:
        for d in range(1, n_draws + 1):
            rng = np.random.default_rng([SEED_BASE, d, int(round(q * 1000)), int(r)])
            col = np.full(len(base), np.nan)
            col[pop] = perturb(needs, q, r, rng)
            draws[d] = col
    columns = {}
    for j in range(1, n_probs + 1):
        for d in range(1, n_draws + 1):
            idx = (j - 1) * n_draws + d
            columns[f"m_{idx}"] = base[f"p_{j}"]
            if q is not None:
                columns[f"exp_need_m_{idx}"] = draws[d]
    return pd.concat([base, pd.DataFrame(columns, index=base.index)], axis=1)


def metric_rows(frame, people, table, algorithm, k, common, n_draws=1):
    """One row per (cohort, model, draw): disparity on true needs, counts,
    utilities, and the RMSE of the needs the DP was given, measured on the
    cohort. Column m_idx maps back to model j = (idx-1)//n_draws + 1 and
    draw d = (idx-1) % n_draws + 1."""
    rows = []
    splits = {seed: frame.loc[(frame["seed"] == seed) & (frame["person_id"] >= 0)]
                        .set_index("person_id")
              for seed in frame["seed"].unique()}
    for _, info in people.iterrows():
        seed, iteration = info["seed"], info["iteration"]
        cohort = splits[seed].loc[info["people"]]
        group = cohort[GROUP_COL].to_numpy()
        need = cohort[NEED_COL].to_numpy(dtype=np.float64)
        minority = int(min((group == 0).sum(), (group == 1).sum()))
        block = table.loc[(table["seed"] == seed) & (table["iteration"] == iteration)]
        for _, allocation in block.iterrows():
            sel = cohort.loc[allocation["selected"]]
            g = sel[GROUP_COL].to_numpy()
            n0, n1 = int((g == 0).sum()), int((g == 1).sum())
            disparity = (abs(sel.loc[g == 1, NEED_COL].mean() - sel.loc[g == 0, NEED_COL].mean())
                         if n0 and n1 else np.nan)
            for model in allocation["models"]:
                idx = int(model.split("_")[1])
                j, draw = (idx - 1) // n_draws + 1, (idx - 1) % n_draws + 1
                rmse = np.nan
                if algorithm == "LDA-exp":
                    given = cohort[f"exp_need_{model}"].to_numpy(dtype=np.float64)
                    rmse = float(np.sqrt(np.mean((given - need) ** 2)))
                rows.append({
                    **common, "algorithm": algorithm, "model": j, "draw": draw,
                    "seed": seed, "iteration": iteration, "k": k,
                    "absolute_disparity": disparity, "k_0": n0, "k_1": n1,
                    "expected_utility": float(sel[f"p_{j}"].mean()),
                    "realized_utility": float(sel["y"].mean()),
                    "cohort_rmse": rmse, "cohort_minority": minority,
                })
    return rows


def summary(df):
    """Mean RMSE and disparity per (q, r), with the sd split the way
    analyse_disparity.py does it: within cohort = over draws, between = over cohorts."""
    out = []
    for (alg, q, r), g in df.groupby(["algorithm", "q", "r"], dropna=False, sort=False):
        per_cohort = g.groupby(["seed", "iteration"])["absolute_disparity"]
        per_model = g.groupby("model")["absolute_disparity"].mean()
        per_draw = g.groupby("draw")["absolute_disparity"].mean()
        out.append({
            "algorithm": alg, "q": q, "r": r,
            "rmse": g["cohort_rmse"].mean(),
            "mean": g["absolute_disparity"].mean(),
            "sd_within": per_cohort.std().mean(),
            "sd_between": per_cohort.mean().std(),
            "sd_models": per_model.std() if len(per_model) > 1 else np.nan,
            "sd_draws": per_draw.std() if len(per_draw) > 1 else np.nan,
            "pct_undefined": 100 * g["absolute_disparity"].isna().mean(),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, required=True)
    ap.add_argument("--selection-rate", type=float, nargs="+", required=True)
    ap.add_argument("--iters-per-seed", type=int, default=20)
    ap.add_argument("--n-draws", type=int, default=25)
    ap.add_argument("--q", type=float, nargs="+", default=DEFAULT_Q)
    ap.add_argument("--r", type=int, nargs="+", default=DEFAULT_R)
    ap.add_argument("--group-split", default="uniform", choices=["uniform", "population"])
    ap.add_argument("--need-threshold", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--pred-method", default="average", choices=["average", "argmax"])
    ap.add_argument("--n-models", type=int, default=1000,
                    help="models in the prediction file to choose the reference from")
    ap.add_argument("--probs-models", type=int, default=1,
                    help="lowest-validation-loss models per split whose probabilities are used")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", default="data/results/perturbation")
    args = ap.parse_args()

    start = time.time()
    fp = ROOT / ("data/predictions/"
                 f"test_bootstrap_{MODEL_CLASS}_{args.pred_method}_qualified_gagne_{args.need_threshold}.csv")
    base, chosen = load_reference(fp, args.n_models, args.probs_models)
    race = pd.read_csv(ROOT / "data/obermeyer/obermeyer_data_cleaned.csv", usecols=["person_id", GROUP_COL])
    base = base.merge(race, on="person_id", how="left")
    base[GROUP_COL] = base[GROUP_COL].fillna(0).astype(int)
    print(f"[start] size={args.sample_size} rates={args.selection_rate} iters={args.iters_per_seed} "
          f"draws={args.n_draws} q={args.q} r={args.r} split={args.group_split} "
          f"probs_models={args.probs_models}", flush=True)
    print("  reference model per split: " +
          ", ".join(f"seed {s}: {ms[0][0]} (val loss {ms[0][1]:.4f})" for s, ms in chosen.items()), flush=True)
    if args.probs_models > 1:
        print("  probability models per split: " +
              "; ".join(f"seed {s}: " + " ".join(m for m, _ in ms) for s, ms in chosen.items()), flush=True)
    M = args.probs_models

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def allocator(frame, rate, n_models):
        return SweepAllocations(
            df=frame, sample_size=args.sample_size, selection_rate=rate,
            qualification_col="y", need_col=NEED_COL, group_col=GROUP_COL,
            rashomon_eps=1.0, allocation_eps=ALLOCATION_EPS, n_models=n_models,
            iters_per_seed=args.iters_per_seed, n_jobs=args.n_jobs,
            show_progress=False, group_split=args.group_split)

    for rate in args.selection_rate:
        t_rate = time.time()
        common = {"sample_size": args.sample_size, "selection_rate": rate,
                  "iters_per_seed": args.iters_per_seed, "group_split": args.group_split,
                  "probs_models": M,
                  "need_threshold": args.need_threshold, "pred_method": args.pred_method}
        ref_frame = model_frame(base, 1, n_probs=M)
        a = allocator(ref_frame, rate, M)
        opt, tru, _, people = a.get_allocations(algorithms=["OPT", "LDA-tru"])
        rows = (metric_rows(ref_frame, people, opt, "OPT", a.k, {**common, "q": np.nan, "r": np.nan})
                + metric_rows(ref_frame, people, tru, "LDA-tru", a.k, {**common, "q": np.nan, "r": np.nan}))
        print(f"  rate {rate}: k={a.k}, {len(people)} cohorts, references done "
              f"at {(time.time()-t_rate)/60:.1f} min", flush=True)

        for q in args.q:
            for r in args.r:
                t0 = time.time()
                frame = model_frame(base, args.n_draws, q, r, n_probs=M)
                a = allocator(frame, rate, M * args.n_draws)
                _, _, exp, people_qr = a.get_allocations(exp_need_col="exp_need_{model}",
                                                         algorithms=["LDA-exp"])
                if not people_qr["people"].equals(people["people"]):
                    sys.exit("cohorts differ between the reference run and a sweep point")
                rows += metric_rows(frame, people_qr, exp, "LDA-exp", a.k, {**common, "q": q, "r": r},
                                    n_draws=args.n_draws)
                print(f"    q={q} r={r}: {(time.time()-t0)/60:.1f} min", flush=True)

        df = pd.DataFrame(rows)
        tag = (f"size-{args.sample_size}_rate-{rate}_iters-{args.iters_per_seed}"
               f"_draws-{args.n_draws}_split-{args.group_split}"
               + (f"_probs-{M}" if M > 1 else "")
               + f"_{args.pred_method}_gagne_{args.need_threshold}")
        df.to_csv(out_dir / f"perturb_{tag}.csv", index=False)
        print(f"  wrote perturb_{tag}.csv ({len(df):,} rows)\n", flush=True)
        with pd.option_context("display.width", 120, "display.float_format", "{:.3f}".format):
            print(summary(df).to_string(index=False), flush=True)
        print(f"  rate {rate} finished in {(time.time()-t_rate)/60:.1f} min\n", flush=True)

    print(f"[done] in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
