"""Run the true-needs allocation experiment for one (selection rate, qualification column).

Faithful port of test/02-experiments_with_tru_needs.ipynb, parametrized so the nine
configurations can run unattended instead of hand-editing cell 1 nine times. Output
paths match the notebook exactly, so results are interchangeable with a notebook run.

    uv run python scripts/run_experiment.py --selection-rate 0.5 --qualification-column qualified_gagne_1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.allocations import Allocations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-rate", type=float, required=True, choices=[0.1, 0.25, 0.5])
    ap.add_argument("--qualification-column", required=True,
                    choices=["qualified_gagne_1", "qualified_gagne_2", "qualified_gagne_3"])
    ap.add_argument("--method", default="test_bootstrap_nn")
    ap.add_argument("--test-size", type=int, default=250)
    ap.add_argument("--need-column", default="gagne_sum_t")
    ap.add_argument("--group-column", default="race")
    ap.add_argument("--rashomon-epsilon", type=float, default=0.01)
    ap.add_argument("--allocation-delta", type=float, default=0.1)
    ap.add_argument("--num-models", type=int, default=100)
    ap.add_argument("--iterations-per-split", type=int, default=10)
    # The notebook never seeds the cohort draw, so repeat runs see different cohorts.
    # Seeding by default makes a run reproducible; pass -1 for the notebook's behaviour.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="exit immediately if the comparison CSV already exists")
    args = ap.parse_args()

    method = args.method
    TEST_SIZE = args.test_size
    SELECTION_RATE = args.selection_rate
    QUALIFICATION_COLUMN = args.qualification_column
    NEED_COLUMN = args.need_column
    GROUP_COLUMN = args.group_column

    tag = f"{method}_y-{QUALIFICATION_COLUMN}_rate-{SELECTION_RATE}_size-{TEST_SIZE}"
    fp_input = ROOT / "data/obermeyer/obermeyer_data_cleaned.csv"
    fp_predictions = ROOT / f"data/predictions/{method}_{QUALIFICATION_COLUMN}.csv"
    fp_output_opt = ROOT / f"data/results/allocations/opt_tru_need_{tag}.csv"
    fp_output_lda = ROOT / f"data/results/allocations/lda_tru_need_{tag}.csv"
    fp_output_people = ROOT / f"data/results/allocations/people_tru_need_{tag}.csv"
    fp_output = ROOT / f"data/results/comparison_tru_need_{tag}.csv"

    if args.skip_existing and fp_output.exists():
        print(f"[skip] {fp_output} already exists", flush=True)
        return

    for fp in (fp_input, fp_predictions):
        if not fp.exists():
            sys.exit(f"missing input: {fp}")

    if args.seed >= 0:
        np.random.seed(args.seed)

    start = time.time()
    print(f"[start] {tag}  num_models={args.num_models}  seed={args.seed}", flush=True)

    data = pd.read_csv(fp_input)
    df = pd.read_csv(fp_predictions)
    df = df.merge(
        data[["person_id", GROUP_COLUMN]],
        on="person_id",
        how="left",
        validate="many_to_one",
    )

    a = Allocations(
        df=df,
        TEST_SIZE=TEST_SIZE,
        SELECTION_RATE=SELECTION_RATE,
        QUALIFICATION_COLUMN="y",
        NEED_COLUMN=NEED_COLUMN,
        GROUP_COLUMN=GROUP_COLUMN,
        RASHOMON_EPSILON=args.rashomon_epsilon,
        ALLOCATION_EPSILON=args.allocation_delta,
        NUM_MODELS=args.num_models,
        ITERATIONS_PER_SPLIT=args.iterations_per_split,
    )
    allocations_opt, allocations_lda, people = a.get_allocations()
    print(f"{len(people)} cohorts, {TEST_SIZE} people per cohort, k = {a.k}", flush=True)

    for fp in (fp_output_opt, fp_output_lda, fp_output_people, fp_output):
        fp.parent.mkdir(parents=True, exist_ok=True)

    allocations_opt.to_csv(fp_output_opt)
    allocations_lda.to_csv(fp_output_lda)
    people.to_csv(fp_output_people)

    all_results = []
    for _, cohort_info in people.iterrows():
        seed = cohort_info["seed"]
        iteration = cohort_info["iteration"]

        split_df = df.loc[(df["seed"] == seed) & (df["person_id"] >= 0)].set_index("person_id")
        cohort = split_df.loc[cohort_info["people"]]

        for algorithm, allocations in [("OPT", allocations_opt), ("LDA", allocations_lda)]:
            cohort_allocations = allocations.loc[
                (allocations["seed"] == seed) & (allocations["iteration"] == iteration)
            ]

            # Count selected sets, not model rows or allocation_idx values.
            unique_allocations = {
                tuple(sorted(selected_ids)) for selected_ids in cohort_allocations["selected"]
            }
            model_names = {model for models in cohort_allocations["models"] for model in models}
            num_unique_allocations = len(unique_allocations)
            num_models = len(model_names)
            unique_allocation_ratio = num_unique_allocations / num_models if num_models else np.nan

            for _, allocation in cohort_allocations.iterrows():
                selected = cohort.loc[allocation["selected"]]

                mean_0 = selected.loc[selected[GROUP_COLUMN] == 0, NEED_COLUMN].mean()
                mean_1 = selected.loc[selected[GROUP_COLUMN] == 1, NEED_COLUMN].mean()
                absolute_disparity = abs(mean_1 - mean_0)
                disparity_ratio = np.nan
                if pd.notna(mean_0) and pd.notna(mean_1) and mean_0 != 0 and mean_1 != 0:
                    disparity_ratio = max(mean_1 / mean_0, mean_0 / mean_1)

                for model in allocation["models"]:
                    all_results.append({
                        "method": method,
                        "algorithm": algorithm,
                        "qualification_column": QUALIFICATION_COLUMN,
                        "seed": seed,
                        "iteration": iteration,
                        "model": model,
                        "k": a.k,
                        "epsilon": args.allocation_delta,
                        "tolerance_type": "add",
                        "expected_utility": selected[model].mean(),
                        "realized_utility": selected["y"].mean(),
                        "absolute_disparity": absolute_disparity,
                        "disparity_ratio": disparity_ratio,
                        "num_models": num_models,
                        "num_unique_allocations": num_unique_allocations,
                        "unique_allocation_ratio": unique_allocation_ratio,
                    })

    results = pd.DataFrame(all_results)
    results.to_csv(fp_output, index=False)

    elapsed = time.time() - start
    print(f"[done] {tag}  {len(results)} rows -> {fp_output}  ({elapsed/60:.1f} min)", flush=True)

    summary = (
        results.groupby("algorithm")[
            ["absolute_disparity", "disparity_ratio", "realized_utility", "num_unique_allocations"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    print(summary.to_string(), flush=True)

    (ROOT / "data/results/runlog.jsonl").open("a").write(
        json.dumps({
            "tag": tag, "elapsed_sec": round(elapsed, 1), "rows": len(results),
            "k": a.k, "num_models": args.num_models, "seed": args.seed,
            "cohorts": len(people),
        }) + "\n"
    )


if __name__ == "__main__":
    main()
