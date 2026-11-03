"""Summary of run_variability.py: for each (n, k) and the first C cohorts per
seed, the mean and both standard deviations of the disparity.

  sd_cohorts: per cohort, mean across models; sd of those cohort means
              (Kshitij's definition; the mean is the mean of the same values).
  sd_models:  per model, mean across cohorts; sd of those model means.

    uv run python scripts/summarise_variability.py [--cohorts 10 25 50 100]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/results/variability")
    ap.add_argument("--cohorts", type=int, nargs="+", default=[10, 25, 50, 100])
    args = ap.parse_args()
    in_dir = ROOT / args.in_dir

    rows = []
    for fp in sorted(glob.glob(str(in_dir / "var_size-*_k-*.csv"))):
        df = pd.read_csv(fp)
        n, k = int(df["sample_size"].iloc[0]), int(df["k"].iloc[0])
        for C in args.cohorts:
            sub = df[df["iteration"] < C]
            for alg, g in sub.groupby("algorithm", sort=False):
                per_cohort = g.groupby(["seed", "iteration"])["absolute_disparity"].mean()
                per_model = g.groupby("model")["absolute_disparity"].mean()
                rows.append({
                    "n": n, "k": k, "cohorts_per_seed": C, "cohorts": per_cohort.notna().sum(),
                    "algorithm": alg,
                    "mean": per_cohort.mean(),
                    "sd_cohorts": per_cohort.std(),
                    "se_mean": per_cohort.std() / np.sqrt(per_cohort.notna().sum()),
                    "sd_models": per_model.std(),
                    "rmse": g["cohort_rmse"].mean(),
                    "pct_undefined": 100 * g["absolute_disparity"].isna().mean(),
                })
    tab = pd.DataFrame(rows)
    out = in_dir / "summary_variability.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out} ({len(tab)} rows)\n")
    with pd.option_context("display.width", 160, "display.max_rows", 400,
                           "display.float_format", "{:.3f}".format):
        for alg in ("LDA-exp", "OPT", "LDA-tru"):
            t = tab[tab.algorithm == alg]
            if t.empty:
                continue
            for value in ("mean", "sd_cohorts", "sd_models"):
                print(f"{alg}: {value}  (rows n, k; columns cohorts per seed)")
                print(t.pivot_table(index=["n", "k"], columns="cohorts_per_seed", values=value).to_string())
                print()


if __name__ == "__main__":
    main()
