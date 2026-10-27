"""Run with python -B -m unittest discover -s test -p 'test_error_before_rounding.py'."""
import itertools
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.allocations import Allocations


def fixture():
    cohort = pd.DataFrame({
        "person_id": range(8), "seed": 0, "race": [0] * 4 + [1] * 4,
        "h": [1, 3, 5, 7, 2, 4, 6, 8], "y": [1, 0, 1, 0, 1, 1, 0, 1],
        "m_1": np.array([30, 25, 18, 14, 29, 24, 20, 12]) / 32,
        "m_2": np.array([28, 26, 20, 15, 30, 23, 19, 11]) / 32,
        "exp_need_m_1": [3.4, 2.25, 2.75, 4.1, 8.2, 0.4, 7.3, 5.6],
        "exp_need_m_2": [1.1, 4.7, 3.3, 6.4, 2.9, 3.6, 5.1, 7.2],
    })
    perf = pd.DataFrame({"person_id": [-1], "m_1": [0.1], "m_2": [0.105]})
    return cohort, perf


def allocator(frame, sample_size=8, iters_per_seed=1):
    return Allocations(frame, sample_size, 0.5, "y", "h", "race", 0.01, 0.125,
                       n_models=2, iters_per_seed=iters_per_seed)


def by_model(rows):
    return {model: tuple(row["selected"]) for row in rows for model in row["models"]}


class BeforeRoundingTests(unittest.TestCase):
    def test_order_integer_representation_and_midpoint_rule(self):
        cohort, perf = fixture()
        cohort.loc[0, "h"] = 2
        a = allocator(cohort)
        values, _ = a._prepare_needs_before_rounding(cohort, perf, "exp_need_{model}", 0.5, 1)
        # 2 + .5 * (3.4 - 2) = 2.7 -> 3, unlike legacy 2 + .5 * (3 - 2) = 2.5.
        self.assertEqual(values["m_1"].iloc[0], 3)
        values, _ = a._prepare_needs_before_rounding(cohort, perf, "exp_need_{model}", 1, 2)
        self.assertEqual(values["m_1"].iloc[1], 4)  # 2.25 -> 2.0
        self.assertEqual(values["m_1"].iloc[2], 6)  # 2.75 -> 3.0
        for step in (1, 0.5, 0.2):
            scale = a._need_grid_scale(0, "before_rounding", step)
            values, diagnostics = a._prepare_needs_before_rounding(cohort, perf, "exp_need_{model}", 0.5, scale)
            for model, needs in values.items():
                self.assertTrue(all(isinstance(value, int) for value in needs))
                adjusted = cohort.h + 0.5 * (cohort[f"exp_need_{model}"] - cohort.h)
                np.testing.assert_allclose(needs.to_numpy(dtype=float) / scale,
                                           np.rint(adjusted * scale) / scale)
            for row in diagnostics:
                for metric in ("mae", "rmse"):
                    self.assertAlmostEqual(row[f"before_rounding_{metric}"], 0.5 * row[f"prediction_{metric}"])
                self.assertLessEqual(row["rounding_mae"], step / 2 + 1e-12)

    def test_endpoints_and_no_input_mutation(self):
        cohort, perf = fixture()
        original = cohort.copy(deep=True)
        a = allocator(cohort)
        truth = a.calculate_least_discriminatory_allocation(0, 0, cohort, perf)
        legacy = a.calculate_least_discriminatory_allocation_exp_needs(0, 0, cohort, perf, "exp_need_{model}")
        for step in (1, 0.5, 0.2):
            actual = a.calculate_least_discriminatory_allocation_exp_needs(
                0, 0, cohort, perf, "exp_need_{model}", beta=0,
                shrinkage_order="before_rounding", need_step=step)
            self.assertEqual(actual, truth)
        actual = a.calculate_least_discriminatory_allocation_exp_needs(
            0, 0, cohort, perf, "exp_need_{model}", beta=1, shrinkage_order="before_rounding")
        self.assertEqual(actual, legacy)
        pd.testing.assert_frame_equal(cohort, original)
        self.assertEqual(len(a.need_diagnostics), 8)

    def test_dp_against_exhaustive_search(self):
        cohort, perf = fixture()
        a = allocator(cohort)
        for beta, step in itertools.product((0, 0.25, 0.5, 0.75, 1), (1, 0.5, 0.2)):
            with self.subTest(beta=beta, step=step):
                scale = a._need_grid_scale(0, "before_rounding", step)
                needs_by_model, _ = a._prepare_needs_before_rounding(cohort, perf, "exp_need_{model}", beta, scale)
                result = a.calculate_least_discriminatory_allocation_exp_needs(
                    0, 0, cohort, perf, "exp_need_{model}", beta=beta,
                    shrinkage_order="before_rounding", need_step=step)
                for model, selected in by_model(result).items():
                    needs = needs_by_model[model]
                    utility_floor = cohort[model].nlargest(a.k).sum() - a.k * a.allocation_eps

                    def objective(indices):
                        groups = [[i for i in indices if cohort.loc[i, "race"] == g] for g in (0, 1)]
                        if not all(groups):
                            return None
                        utility = cohort.loc[list(indices), model].sum()
                        if utility < utility_floor:
                            return None
                        gap = abs(Fraction(sum(needs.loc[groups[0]]), scale * len(groups[0]))
                                  - Fraction(sum(needs.loc[groups[1]]), scale * len(groups[1])))
                        return gap, -utility

                    possibilities = [value for indices in itertools.combinations(cohort.index, a.k)
                                     if (value := objective(indices)) is not None]
                    self.assertEqual(objective(selected), min(possibilities))

    def test_existing_alpha_and_beta_formula(self):
        cohort, perf = fixture()
        a = allocator(cohort)
        for alpha, beta in itertools.product((0, 0.5, 1), (0, 0.25, 0.5, 1)):
            scale = Fraction(str(beta)).denominator
            numerator = Fraction(str(beta)).numerator
            indices = a._true_need_indices(cohort, 0, 0, alpha, 17)
            needs = {}
            for model in a.rashomon_models(perf):
                mixed = (scale - numerator) * cohort.h + numerator * np.rint(cohort[f"exp_need_{model}"])
                mixed.loc[indices] = scale * cohort.loc[indices, "h"]
                needs[model] = mixed.astype(int)
            expected = a.calculate_least_discriminatory_allocation(0, 0, cohort, perf, needs_by_model=needs, need_scale=scale)
            actual = a.calculate_least_discriminatory_allocation_exp_needs(
                0, 0, cohort, perf, "exp_need_{model}", alpha=alpha, beta=beta, replacement_seed=17)
            self.assertEqual(by_model(actual), by_model(expected))
        self.assertEqual(a.need_diagnostics, [])

    def test_get_allocations_cohorts_coverage_and_diagnostic_reset(self):
        cohort, perf = fixture()
        expanded = pd.concat([cohort.assign(person_id=cohort.person_id + 8 * i) for i in range(4)], ignore_index=True)
        frames = [pd.concat([expanded.assign(seed=seed), perf.assign(seed=seed)], ignore_index=True) for seed in (0, 1)]
        df = pd.concat(frames, ignore_index=True)
        a = allocator(df, iters_per_seed=2)
        first_people = None
        for beta, step in ((0.25, 1), (0.5, 0.5), (1, 0.2)):
            opt, tru, exp, people = a.get_allocations(
                "exp_need_{model}", beta=beta, shrinkage_order="before_rounding", need_step=step,
                algorithms="LDA-exp")
            self.assertTrue(opt.empty and tru.empty)
            self.assertFalse(exp.empty)
            self.assertEqual(len(a.need_diagnostics), 8)
            self.assertTrue(people.need_scale.eq(round(1 / step)).all())
            self.assertTrue(people.shrinkage_order.eq("before_rounding").all())
            identity = people[["seed", "iteration", "people"]]
            if first_people is None:
                first_people = identity
            else:
                pd.testing.assert_frame_equal(identity, first_people)
        a.get_allocations("exp_need_{model}", algorithms="OPT")
        self.assertEqual(a.need_diagnostics, [])

    def test_invalid_parameters_and_nonfinite_predictions(self):
        cohort, perf = fixture()
        a = allocator(cohort)
        for step in (0, -1, 2, 0.3, np.nan, np.inf, "0.5", True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                a.get_allocations("exp_need_{model}", shrinkage_order="before_rounding", need_step=step)
        for options in ({"shrinkage_order": "unknown"}, {"need_step": 0.5},
                        {"shrinkage_order": "before_rounding", "alpha": 0.5},
                        {"shrinkage_order": "before_rounding", "algorithms": "OPT"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                a.get_allocations("exp_need_{model}", **options)
        cohort.loc[0, "exp_need_m_1"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            a.calculate_least_discriminatory_allocation_exp_needs(
                0, 0, cohort, perf, "exp_need_{model}", shrinkage_order="before_rounding")
        cohort.loc[0, "exp_need_m_1"] = 2
        cohort["h"] = cohort["h"].astype(float)
        cohort.loc[0, "h"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer true needs"):
            a.calculate_least_discriminatory_allocation_exp_needs(
                0, 0, cohort, perf, "exp_need_{model}", shrinkage_order="before_rounding")


if __name__ == "__main__":
    unittest.main()
