"""Diversity and homogenization metrics, computed from saved allocations.

Every metric here is derived from the files already in data/results/allocations/,
so adding one never means re-running the DP.

Allocations are expanded per model before anything is measured: a row in the
allocation files is one *distinct* selection, and its `models` column lists the
models that produced it, so a selection found by 40 models counts 40 times. That
matches "two random allocations from the set" and lets p be read as a per-person
selection probability.

    uv run python scripts/compute_metrics.py                 # every config found
    uv run python scripts/compute_metrics.py --self-test     # check the maths
"""

import argparse
import ast
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def binary_entropy(p):
    """Shannon entropy in bits of a Bernoulli(p), vectorised, 0 at p in {0,1}."""
    p = np.clip(p, 0.0, 1.0)
    out = np.zeros_like(p, dtype=float)
    m = (p > 0) & (p < 1)
    q = p[m]
    out[m] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return out


def cohort_metrics(selected_lists, people_ids, qualified_mask):
    """Metrics for one cohort, given one selection per model."""
    n = len(people_ids)
    pos = {pid: j for j, pid in enumerate(people_ids)}
    m = len(selected_lists)

    B = np.zeros((m, n), dtype=np.float64)
    for i, sel in enumerate(selected_lists):
        B[i, [pos[p] for p in sel]] = 1.0
    k = int(B[0].sum())

    # Pairwise quantities from one matrix multiply instead of m^2 comparisons.
    inter = B @ B.T
    iu = np.triu_indices(m, k=1)
    if len(iu[0]):
        pair_inter = inter[iu]
        # symmetric difference = 2k - 2*inter, scaled to [0, 1]
        pairwise_difference = float(np.mean(1.0 - pair_inter / k))
        # share of people given the same outcome by both allocations
        pairwise_consistency = float(np.mean((n - 2 * k + 2 * pair_inter) / n))
    else:
        pairwise_difference, pairwise_consistency = 0.0, 1.0

    p = B.mean(axis=0)
    never = p == 0
    return {
        "num_models": m,
        "num_unique_allocations": len({tuple(sorted(s)) for s in selected_lists}),
        "k": k,
        "pairwise_difference": pairwise_difference,
        "pairwise_consistency": pairwise_consistency,
        "heterozygosity": float(np.mean(2 * p * (1 - p))),
        "selection_entropy": float(np.mean(binary_entropy(p))),
        "share_chance_ge_10pct": float(np.mean(p >= 0.10)),
        "systemic_exclusion_all": float(np.mean(never)),
        "systemic_exclusion_qualified": (
            float(np.mean(never[qualified_mask])) if qualified_mask.any() else np.nan
        ),
    }


def run_config(people_fp, qual_column_cache):
    tag = Path(people_fp).name[len("people_tru_need_"):-len(".csv")]
    opt_fp = Path(people_fp).with_name(f"opt_tru_need_{tag}.csv")
    lda_tru_fp = Path(people_fp).with_name(f"lda_tru_need_{tag}.csv")
    lda_exp_fp = Path(people_fp).with_name(f"lda_exp_need_{tag}.csv")
    if not (opt_fp.exists() and lda_tru_fp.exists() and lda_exp_fp.exists()):
        print(f"  skip {tag}: missing opt/lda files", flush=True)
        return None

    qual = re.search(r"y-(qualified_gagne_\d)", tag).group(1)
    if qual not in qual_column_cache:
        pred = pd.read_csv(ROOT / f"data/predictions/test_bootstrap_nn_{qual}.csv",
                           usecols=["person_id", "seed", "y"])
        pred = pred[pred["person_id"] >= 0]
        qual_column_cache[qual] = {
            s: dict(zip(g["person_id"], g["y"])) for s, g in pred.groupby("seed")
        }
    y_by_seed = qual_column_cache[qual]

    people = pd.read_csv(people_fp)
    people["people"] = people["people"].apply(ast.literal_eval)

    rows = []
    for algorithm, fp in (("OPT", opt_fp), ("LDA-tru", lda_tru_fp), ("LDA-exp", lda_exp_fp)):
        print(algorithm, fp)
        alloc = pd.read_csv(fp)
        alloc["selected"] = alloc["selected"].apply(ast.literal_eval)
        alloc["models"] = alloc["models"].apply(ast.literal_eval)
        grouped = {key: g for key, g in alloc.groupby(["seed", "iteration"])}

        for r in people.itertuples():
            key = (r.seed, r.iteration)
            if key not in grouped:
                continue
            g = grouped[key]
            # one entry per model, so a selection found by many models counts many times
            selected_lists = [sel for sel, ms in zip(g["selected"], g["models"]) for _ in ms]
            ymap = y_by_seed[r.seed]
            qualified_mask = np.array([ymap.get(p, 0) == 1 for p in r.people])
            rows.append({
                "tag": tag, "algorithm": algorithm, "seed": r.seed, "iteration": r.iteration,
                **cohort_metrics(selected_lists, r.people, qualified_mask),
            })

    out = pd.DataFrame(rows)
    fp_out = ROOT / f"data/results/diversity_tru_need_{tag}.csv"
    out.to_csv(fp_out, index=False)
    return out


def self_test():
    """Degenerate cases where the right answer is known by hand."""
    people = list(range(10))
    qualified = np.array([True] * 5 + [False] * 5)

    same = [[0, 1, 2]] * 8
    m = cohort_metrics(same, people, qualified)
    assert abs(m["pairwise_difference"]) < 1e-12, m
    assert abs(m["selection_entropy"]) < 1e-12, m
    assert abs(m["heterozygosity"]) < 1e-12, m
    assert abs(m["share_chance_ge_10pct"] - 3 / 10) < 1e-12, m
    assert abs(m["systemic_exclusion_all"] - 7 / 10) < 1e-12, m
    assert m["num_unique_allocations"] == 1, m

    disjoint = [[0, 1, 2], [3, 4, 5]]
    m = cohort_metrics(disjoint, people, qualified)
    assert abs(m["pairwise_difference"] - 1.0) < 1e-12, m
    assert abs(m["pairwise_consistency"] - 4 / 10) < 1e-12, m

    half = [[0, 1, 2], [0, 1, 3]]
    m = cohort_metrics(half, people, qualified)
    assert abs(m["pairwise_difference"] - 1 / 3) < 1e-12, m
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pattern", default="people_tru_need_*.csv")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    files = sorted(glob.glob(str(ROOT / "data/results/allocs" / args.pattern)))
    if not files:
        sys.exit("no people_*.csv files found")

    cache, frames = {}, []
    for fp in files:
        print(f"computing {Path(fp).name}", flush=True)
        out = run_config(fp, cache)
        if out is not None:
            frames.append(out)

    allm = pd.concat(frames, ignore_index=True)
    allm.to_csv(ROOT / "data/results/diversity_all.csv", index=False)
    cols = ["num_unique_allocations", "pairwise_difference", "pairwise_consistency",
            "heterozygosity", "selection_entropy", "share_chance_ge_10pct",
            "systemic_exclusion_all", "systemic_exclusion_qualified"]
    print("\n=== mean by config and algorithm ===")
    print(allm.groupby(["tag", "algorithm"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()