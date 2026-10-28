"""Decompose the allocation disparity results from run_allocations.py.

A pooled standard deviation over the comparison files mixes two unrelated
things: the spread across the Rashomon set inside one cohort, which is the
multiplicity the paper is about, and the spread between cohorts, which is the
cohort sampler. This script separates them, and reports two measurement
caveats that affect how the OPT column should be read.

  * Top-k selects nobody from one group in a fifth of rows at the lowest
    selection rate. absolute_disparity is undefined there and those rows drop
    out of OPT's mean, so OPT is averaged over a different, more balanced
    subset than the DP rows it is compared against.
  * The DP's group loop runs count_0 from max(1, k - n_1) to k - 1, so it
    always takes at least one person from each group. Top-k has no such
    constraint. Part of the OPT/LDA gap is that, not the objective.

Both follow from the cohort sampler in allocations_parallel.py drawing
count_0 = rng.choice(range(1, sample_size)), a uniform group split, which
leaves some cohorts with almost nobody in one group.

Writes data/results/disparity_sd_decomposition.csv,
data/results/disparity_by_cohort_balance.csv, and, when matplotlib is
importable, a summary table as a PNG:

    uv run python scripts/analyse_disparity.py
    uv run --no-project --with matplotlib --with pandas \
        python -I scripts/analyse_disparity.py --png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_CLASS = "nn_multiclass"
ALGORITHMS = ("OPT", "LDA-tru", "LDA-exp")
THRESHOLDS = (1, 2, 3)
RATES = ("0.1", "0.25", "0.5")
SAMPLE_SIZE = 100
ITERS_PER_SEED = 10
N_SEEDS = 5


def cohort_composition(n_seeds=N_SEEDS, iters=ITERS_PER_SEED, sample_size=SAMPLE_SIZE):
    """Reproduce the group split allocations_parallel.py draws per cohort.

    The allocator creates one default_rng(0) and advances it once per cohort,
    over seeds in sorted order, so the draws replay exactly.
    """
    rng = np.random.default_rng(0)
    draws = [int(rng.choice(range(1, sample_size))) for _ in range(n_seeds * iters)]
    out = pd.DataFrame({
        "seed": np.repeat(range(n_seeds), iters),
        "iteration": list(range(iters)) * n_seeds,
        "count_0": draws,
    })
    out["count_1"] = sample_size - out["count_0"]
    out["minority"] = out[["count_0", "count_1"]].min(axis=1)
    return out


def _stem(pred_method, threshold, rate, size, iters, n_models, eps):
    return (f"test_bootstrap_{MODEL_CLASS}_{pred_method}_qualified_gagne_{threshold}"
            f"_rate-{rate}_size-{size}_iters-{iters}_models-{n_models}_eps-{eps}")


def collect(pred_method, size, iters, n_models, eps):
    comp = cohort_composition()
    sd_rows, balance_rows = [], []
    for threshold in THRESHOLDS:
        for rate in RATES:
            k = int(float(rate) * SAMPLE_SIZE)
            stem = _stem(pred_method, threshold, rate, size, iters, n_models, eps)
            fp_c = ROOT / f"data/results/comparison_{stem}.csv"
            fp_d = ROOT / f"data/results/diagnostics_{stem}.csv"
            if not fp_c.exists():
                print(f"  skip {stem}: no comparison file")
                continue
            comparison = pd.read_csv(
                fp_c, usecols=["algorithm", "seed", "iteration", "model",
                               "absolute_disparity", "num_rashomon_models"])
            comparison = comparison.merge(comp, on=["seed", "iteration"])

            excluded = {}
            if fp_d.exists():
                diag = pd.read_csv(fp_d, usecols=["seed", "iteration", "model", "algorithm",
                                                  "selected_count_0", "selected_count_1"])
                empty = (diag["selected_count_0"] == 0) | (diag["selected_count_1"] == 0)
                excluded = diag.assign(empty=empty).groupby("algorithm")["empty"].mean().to_dict()

            for algorithm in ALGORITHMS:
                rows = comparison[comparison["algorithm"] == algorithm]
                if rows.empty:
                    continue
                per_cohort = rows.groupby(["seed", "iteration"])["absolute_disparity"]
                sd_rows.append({
                    "pred_method": pred_method,
                    "need_threshold": threshold,
                    "selection_rate": float(rate),
                    "k": k,
                    "algorithm": algorithm,
                    "n_rashomon": round(rows["num_rashomon_models"].mean(), 1),
                    "mean": rows["absolute_disparity"].mean(),
                    "sd_pooled": rows["absolute_disparity"].std(),
                    "sd_within_cohort": per_cohort.std().mean(),
                    "sd_between_cohort": per_cohort.mean().std(),
                    "pct_undefined": 100 * rows["absolute_disparity"].isna().mean(),
                    "pct_group_excluded": 100 * excluded.get(algorithm, float("nan")),
                })

                balanced = rows[rows["minority"] >= k]
                degenerate = rows[rows["minority"] < k]
                balance_rows.append({
                    "pred_method": pred_method,
                    "need_threshold": threshold,
                    "selection_rate": float(rate),
                    "k": k,
                    "algorithm": algorithm,
                    "n_cohorts_balanced": balanced.groupby(["seed", "iteration"]).ngroups,
                    "n_cohorts_degenerate": degenerate.groupby(["seed", "iteration"]).ngroups,
                    "mean_all": rows["absolute_disparity"].mean(),
                    "mean_balanced": balanced["absolute_disparity"].mean(),
                    "sd_balanced": balanced["absolute_disparity"].std(),
                    "mean_degenerate": degenerate["absolute_disparity"].mean(),
                })
    return pd.DataFrame(sd_rows), pd.DataFrame(balance_rows)


def text_table(sd, brackets=True):
    """The results as a monospace table, ready to paste into a Slack code block."""
    sd = sd[sd.pred_method == "average"]
    if brackets:
        head = ("thr  rate          OPT                  LDA-tru                LDA-exp\n"
                "---  ----  ---------------------  ---------------------  ---------------------")
        width, cell = 21, (lambda r: f"{r['mean']:.3f} ± {r.sd_pooled:.3f} "
                                     f"({r.sd_within_cohort:.3f})")
    else:
        head = ("thr  rate        OPT            LDA-tru          LDA-exp\n"
                "---  ----  --------------  --------------  --------------")
        width, cell = 14, (lambda r: f"{r['mean']:.3f} ± {r.sd_pooled:.3f}")

    lines = head.split("\n")
    for (threshold, rate), grp in sd.groupby(["need_threshold", "selection_rate"], sort=True):
        cells = [cell(grp[grp.algorithm == a].iloc[0]) for a in ALGORITHMS]
        lines.append(f" {int(threshold)}   {str(rate):<4}  "
                     + "  ".join(f"{c:<{width}}" for c in cells))
    return "\n".join(line.rstrip() for line in lines)


def _style(tb, hl=None, head="#33415a"):
    for (r, col), cell in tb.get_celld().items():
        cell.set_edgecolor("#d8d8d8")
        cell.set_linewidth(.6)
        if r == 0:
            cell.set_facecolor(head)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f5f6f8")
            if hl and col in hl:
                cell.set_facecolor("#eef4ec" if r % 2 else "#e4ede1")
        if col < 2:
            cell.set_text_props(weight="bold" if col == 0 else "normal")


def write_simple_png(sd, fp_png):
    """The headline table: mean +/- sd per algorithm, all three thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sd = sd[sd.pred_method == "average"]
    fig = plt.figure(figsize=(12.6, 5.45), dpi=170)
    fig.patch.set_facecolor("white")
    fig.text(.5, .955, "Absolute disparity, bootstrap, 1000 models, (100,) network",
             ha="center", size=15.5, weight="bold")
    fig.text(.5, .905,
             "mean +/- sd over 5 seeds x 10 cohorts x ~995 Rashomon models   |   "
             "(within) = sd across models inside one cohort",
             ha="center", size=10, color="#555")

    cols = ["thr", "rate"]
    for algorithm in ALGORITHMS:
        cols += [f"{algorithm}", f"{algorithm}\n(within)"]
    cells = []
    for keys, grp in sd.groupby(["need_threshold", "selection_rate"], sort=True):
        row = [int(keys[0]), keys[1]]
        for algorithm in ALGORITHMS:
            a = grp[grp.algorithm == algorithm].iloc[0]
            row += [f"{a['mean']:.3f}  ± {a.sd_pooled:.3f}",
                    f"{a.sd_within_cohort:.3f}"]
        cells.append(row)

    ax = fig.add_axes([.03, .40, .94, .44])
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=cols, cellLoc="center",
                  loc="upper center",
                  colWidths=[.052, .058] + [.168, .082] * 3)
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.4)
    tb.scale(1, 1.62)
    _style(tb, hl=[7])

    fig.canvas.draw()
    bb = tb.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    fig.text(.03, bb.y0 - .035,
             "Most of the pooled sd is variation between cohorts, not between models. The (within) "
             "columns hold the cohort fixed and vary\n"
             "the model across the Rashomon set, so those are the multiplicity. For LDA-tru it is near "
             "zero everywhere: given a cohort, all\n"
             "~995 models produce almost the same disparity.\n\n"
             "Top-k selects nobody from one group in 20.1% of rows at rate 0.1, where the disparity is "
             "undefined and those rows drop out of\n"
             "OPT's mean. The DP always keeps both groups represented, so OPT and the DP are not "
             "averaged over the same allocations.",
             size=9.5, color="#444", va="top")
    fig.savefig(fp_png, bbox_inches="tight", facecolor="white")
    print(f"  wrote {fp_png.name}")


def write_png(sd, balance, comp, fp_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sd = sd[sd.pred_method == "average"]
    balance = balance[balance.pred_method == "average"]
    index = ["need_threshold", "selection_rate"]

    fig = plt.figure(figsize=(15.5, 8.6), dpi=170)
    fig.patch.set_facecolor("white")
    fig.text(.5, .965, "Bootstrap, 1000 models, hidden_layer_sizes=(100,)",
             ha="center", size=17, weight="bold")
    fig.text(.5, .935, "absolute disparity, pred_method = average, Rashomon eps = 0.01, "
             f"{N_SEEDS} seeds x {ITERS_PER_SEED} cohorts of {SAMPLE_SIZE}",
             ha="center", size=10.5, color="#555")

    def table(ax, cols, cells, widths, title, hl=None):
        ax.axis("off")
        ax.set_title(title, loc="left", size=12.5, weight="bold", pad=11)
        tb = ax.table(cellText=cells, colLabels=cols, cellLoc="center",
                      loc="upper center", colWidths=widths)
        tb.auto_set_font_size(False)
        tb.set_fontsize(9.6)
        tb.scale(1, 1.52)
        for (r, col), cell in tb.get_celld().items():
            cell.set_edgecolor("#d8d8d8")
            cell.set_linewidth(.6)
            if r == 0:
                cell.set_facecolor("#33415a")
                cell.set_text_props(color="white", weight="bold")
                cell.set_height(cell.get_height() * 1.25)
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f5f6f8")
                if hl and col in hl:
                    cell.set_facecolor("#eef4ec" if r % 2 else "#e4ede1")
            if col < 2:
                cell.set_text_props(weight="bold" if col == 0 else "normal")
        return tb

    cols1 = ["thr", "rate"]
    for algorithm in ALGORITHMS:
        cols1 += [f"{algorithm}\nmean", f"{algorithm}\nsd",
                  f"{algorithm}\nsd within", f"{algorithm}\nsd between"]
    cells1 = []
    for keys, grp in sd.groupby(index, sort=True):
        row = [int(keys[0]), keys[1]]
        for algorithm in ALGORITHMS:
            a = grp[grp.algorithm == algorithm].iloc[0]
            row += [f"{a['mean']:.3f}", f"{a.sd_pooled:.3f}",
                    f"{a.sd_within_cohort:.3f}", f"{a.sd_between_cohort:.3f}"]
        cells1.append(row)
    ax1 = fig.add_axes([.035, .635, .93, .255])
    tb1 = table(ax1, cols1, cells1, [.042, .048] + [.0755] * 12,
                "1.  Mean and standard deviation, and what the spread is made of",
                hl=[8, 9])
    note1 = (
        "sd          pooled over all models, cohorts and seeds -- the number usually quoted.\n"
        "sd within   spread across the Rashomon models inside one cohort, averaged over cohorts. This is the multiplicity.\n"
        "sd between  spread of cohort means. This is cohort composition, not model choice.\n"
        "For LDA-tru the within term is near zero everywhere: given a cohort, every model in the Rashomon set produces almost the same disparity.")

    cols2 = ["thr", "rate", "cohorts\nof 50", *ALGORITHMS]
    cells2 = []
    for keys, grp in balance.groupby(index, sort=True):
        a = {r.algorithm: r for _, r in grp.iterrows()}
        cells2.append([int(keys[0]), keys[1], int(a["OPT"].n_cohorts_balanced)]
                      + [f"{a[x].mean_balanced:.3f}" for x in ALGORITHMS])
    ax2 = fig.add_axes([.035, .175, .60, .255])
    tb2 = table(ax2, cols2, cells2, [.085, .1, .135, .17, .17, .17],
                "2.  Same metric, restricted to cohorts whose smaller group has at least k people",
                hl=[4])
    note2 = (
        "LDA-tru reaches near-exact need parity once both groups can actually fill the k slots.\n"
        "Note the 'cohorts of 50' column at rate 0.5: only 1 of 50 cohorts qualifies, which is why\n"
        "rate 0.5 behaves differently from the other two throughout.")

    excl = sd[(sd.algorithm == "OPT") & (sd.need_threshold == 1)].set_index("selection_rate")
    pct = " / ".join(f"{excl.loc[float(r), 'pct_group_excluded']:.1f}%" for r in RATES)
    under10 = int((comp.minority < 10).sum())
    under5 = int((comp.minority < 5).sum())
    ax3 = fig.add_axes([.665, .175, .30, .255])
    ax3.axis("off")
    ax3.set_title("3.  Two things to know about the numbers above",
                  loc="left", size=12.5, weight="bold", pad=11)
    ax3.text(0, .97,
             f"a)  Top-k selects nobody from one group in\n"
             f"     {pct} of rows (rate 0.1 /\n"
             "     0.25 / 0.5, threshold 1). Disparity is\n"
             "     undefined there and those rows drop out\n"
             "     of OPT's mean, so OPT is measured on a\n"
             "     different, more balanced subset than the\n"
             "     DP rows it is compared against.\n\n"
             "b)  The DP loop runs count_0 from max(1, ...)\n"
             "     to k-1, so it ALWAYS takes at least one\n"
             "     person from each group. Top-k has no such\n"
             "     constraint. Part of the OPT vs LDA gap is\n"
             "     this structural difference, not the\n"
             "     objective.\n\n"
             "Both follow from the cohort sampler drawing\n"
             f"count_0 ~ Uniform(1, {SAMPLE_SIZE - 1}): {under10} of {len(comp)} cohorts have\n"
             f"a group with fewer than 10 people, {under5} have fewer\n"
             "than 5.",
             transform=ax3.transAxes, size=9.2, va="top",
             family="DejaVu Sans Mono", color="#222")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def below(tb, text, gap=.018):
        bb = tb.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.text(.035, bb.y0 - gap, text, size=9.3, color="#444", va="top")
        return bb.y0 - gap

    y = min(below(tb1, note1), below(tb2, note2))
    n = sd.groupby("need_threshold")["n_rashomon"].max().round().astype(int).tolist()
    fig.text(.035, y - .105,
             f"Rashomon set: {' / '.join(str(x) for x in n)} of 1000 models at thresholds 1 / 2 / 3. "
             "Validation-loss spread 0.011-0.012, against 0.17-0.19 for (50,100,200).",
             size=9.4, color="#444", va="top")
    fig.savefig(fp_png, bbox_inches="tight", facecolor="white")
    print(f"  wrote {fp_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-method", default="average", choices=["average", "argmax", "both"])
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--iters-per-seed", type=int, default=ITERS_PER_SEED)
    ap.add_argument("--n-models", type=int, default=1000)
    ap.add_argument("--rashomon-eps", default="0.01")
    ap.add_argument("--png", action="store_true", help="also render the summary table")
    ap.add_argument("--text", action="store_true",
                    help="print the results table and exit, without rewriting the CSVs")
    ap.add_argument("--no-brackets", action="store_true",
                    help="with --text, leave out the within-cohort sd (57 chars instead of 78)")
    args = ap.parse_args()

    fp_cached = ROOT / "data/results/disparity_sd_decomposition.csv"
    if args.text and fp_cached.exists():
        print(text_table(pd.read_csv(fp_cached), brackets=not args.no_brackets))
        return

    methods = ("average", "argmax") if args.pred_method == "both" else (args.pred_method,)
    sd_all, balance_all = [], []
    for method in methods:
        sd, balance = collect(method, args.sample_size, args.iters_per_seed,
                              args.n_models, args.rashomon_eps)
        sd_all.append(sd)
        balance_all.append(balance)
    sd = pd.concat(sd_all, ignore_index=True)
    balance = pd.concat(balance_all, ignore_index=True)
    if sd.empty:
        raise SystemExit("no comparison files matched; check --n-models and --rashomon-eps")

    out = ROOT / "data/results"
    out.mkdir(parents=True, exist_ok=True)
    fp_sd = out / "disparity_sd_decomposition.csv"
    fp_bal = out / "disparity_by_cohort_balance.csv"
    sd.round(6).to_csv(fp_sd, index=False)
    balance.round(6).to_csv(fp_bal, index=False)
    print(f"  wrote {fp_sd.name}  rows={len(sd)}")
    print(f"  wrote {fp_bal.name}  rows={len(balance)}")

    if args.text:
        print()
        print(text_table(sd, brackets=not args.no_brackets))
        return

    if args.png:
        share = out.parent / "share"
        share.mkdir(parents=True, exist_ok=True)
        write_simple_png(sd, share / "results_table_net100.png")
        write_png(sd, balance, cohort_composition(iters=args.iters_per_seed,
                                                  sample_size=args.sample_size),
                  share / "disparity_table_net100.png")


if __name__ == "__main__":
    main()
