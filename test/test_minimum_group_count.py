"""Run with python -B -m unittest discover -s test -p 'test_minimum_group_count.py'."""
import itertools
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

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


def allocator(frame, sample_size=8, selection_rate=0.5, tolerance=0.125):
    return Allocations(frame, sample_size, selection_rate, "y", "h", "race", 0.01,
                       tolerance, n_models=2, iters_per_seed=2)


def by_model(rows):
    return {model: tuple(row["selected"]) for row in rows for model in row["models"]}


class MinimumGroupCountTests(unittest.TestCase):
    def test_constrained_dp_matches_exhaustive_search(self):
        cohort, perf = fixture()
        for rate, minimum, beta, step in itertools.product((0.5, 0.75), (1, 2, 3), (0, 0.5, 1), (1, 0.2)):
            a = allocator(cohort, selection_rate=rate)
            if 2 * minimum > a.k:
                continue
            with self.subTest(k=a.k, minimum=minimum, beta=beta, step=step):
                scale = round(1 / step)
                needs, _ = a._prepare_needs_before_rounding(cohort, perf, "exp_need_{model}", beta, scale)
                rows = a.calculate_least_discriminatory_allocation_exp_needs(
                    0, 0, cohort, perf, "exp_need_{model}", beta=beta,
                    shrinkage_order="before_rounding", need_step=step, min_selected_per_group=minimum)
                diag = {r['model']: r for r in a.group_count_diagnostics}
                for model, selected in by_model(rows).items():
                    floor = cohort[model].nlargest(a.k).sum() - a.k * a.allocation_eps

                    def objective(indices, min_count):
                        groups = [[i for i in indices if cohort.loc[i, 'race'] == g] for g in (0, 1)]
                        utility = cohort.loc[list(indices), model].sum()
                        if min(map(len, groups)) < min_count or utility < floor:
                            return None
                        gap = abs(Fraction(sum(needs[model].loc[groups[0]]), scale * len(groups[0]))
                                  - Fraction(sum(needs[model].loc[groups[1]]), scale * len(groups[1])))
                        return gap, -utility

                    feasible = [value for ix in itertools.combinations(cohort.index, a.k)
                                if (value := objective(ix, minimum)) is not None]
                    self.assertEqual(diag[model]['constraint_feasible'], bool(feasible))
                    if feasible:
                        self.assertEqual(objective(selected, minimum)[0], min(feasible)[0])
                        # Preserve the existing floating-point disparity tie behavior.
                        # Mathematically equal fractions can differ at the last bit
                        # when the legacy path subtracts two floating-point means.
                        def floating_objective(indices):
                            g0 = [i for i in indices if cohort.loc[i, 'race'] == 0]
                            g1 = [i for i in indices if cohort.loc[i, 'race'] == 1]
                            s0, s1 = sum(needs[model].loc[g0]), sum(needs[model].loc[g1])
                            if beta == 0 or scale == 1:
                                gap = abs((s1 // scale if beta == 0 else s1) / len(g1)
                                          - (s0 // scale if beta == 0 else s0) / len(g0))
                            else:
                                gap = abs(s1 * len(g0) - s0 * len(g1)) / (scale * len(g0) * len(g1))
                            return gap, -cohort.loc[list(indices), model].sum()
                        float_feasible = [floating_objective(ix)
                                          for ix in itertools.combinations(cohort.index, a.k)
                                          if objective(ix, minimum) is not None]
                        self.assertEqual(floating_objective(selected), min(float_feasible))
                    else:
                        baseline = a.calculate_least_discriminatory_allocation_exp_needs(
                            0, 0, cohort, perf, 'exp_need_{model}', beta=beta,
                            shrinkage_order='before_rounding', need_step=step)
                        self.assertEqual(selected, by_model(baseline)[model])

    def test_fallback_for_group_shortage_and_utility(self):
        for shortage in (False, True):
            with self.subTest(shortage=shortage):
                cohort, perf = fixture()
                if shortage:
                    cohort['race'] = [0] * 7 + [1]
                    tolerance = 1
                else:
                    cohort['m_1'] = cohort['m_2'] = np.array([32, 31, 30, 29, 4, 3, 2, 1]) / 32
                    tolerance = 0.25
                a = allocator(cohort, tolerance=tolerance)
                baseline = a.calculate_least_discriminatory_allocation_exp_needs(0, 0, cohort, perf, 'exp_need_{model}')
                a.group_count_diagnostics = []
                with patch.object(a, 'make_group_tables', wraps=a.make_group_tables) as tables:
                    actual = a.calculate_least_discriminatory_allocation_exp_needs(
                        0, 0, cohort, perf, 'exp_need_{model}', min_selected_per_group=2)
                    self.assertEqual(tables.call_count, 4)  # Two groups per model, even on fallback.
                self.assertEqual(actual, baseline)
                self.assertEqual(len(a.group_count_diagnostics), 2)
                for row in a.group_count_diagnostics:
                    self.assertFalse(row['constraint_feasible'])
                    self.assertTrue(row['fallback_used'] and row['baseline_feasible'])
                    self.assertEqual(row['selection_source'], 'baseline_dp')
                    self.assertEqual(row['fallback_reason'], 'insufficient_group_members' if shortage else 'utility_requirement')

    def test_original_topk_fallback_remains_visible(self):
        cohort, perf = fixture()
        cohort['m_1'] = cohort['m_2'] = np.array([32, 31, 30, 29, 4, 3, 2, 1]) / 32
        a = allocator(cohort, tolerance=0)
        rows = a.calculate_least_discriminatory_allocation_exp_needs(
            0, 0, cohort, perf, 'exp_need_{model}', min_selected_per_group=2)
        self.assertEqual(set(by_model(rows).values()), {(0, 1, 2, 3)})
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(a.group_count_diagnostics), 2)  # Do not merge model diagnostics.
        for row in a.group_count_diagnostics:
            self.assertFalse(row['constraint_feasible'] or row['baseline_feasible'])
            self.assertTrue(row['fallback_used'])
            self.assertEqual(row['selection_source'], 'top_k')
            self.assertEqual(row['selected_count_1'], 0)

    def test_alpha_beta_and_default_compatibility(self):
        cohort, perf = fixture()
        original = cohort.copy(deep=True)
        a = allocator(cohort)
        for alpha, beta in itertools.product((0, 0.5, 1), (0, 0.25, 1)):
            scale = Fraction(str(beta)).denominator
            numerator = Fraction(str(beta)).numerator
            replaced = a._true_need_indices(cohort, 0, 0, alpha, 17)
            needs = {}
            for model in a.rashomon_models(perf):
                mixed = (scale - numerator) * cohort.h + numerator * np.rint(cohort[f'exp_need_{model}'])
                mixed.loc[replaced] = scale * cohort.loc[replaced, 'h']
                needs[model] = mixed.astype(int)
            expected = a.calculate_least_discriminatory_allocation(
                0, 0, cohort, perf, needs_by_model=needs, need_scale=scale, min_selected_per_group=2)
            actual = a.calculate_least_discriminatory_allocation_exp_needs(
                0, 0, cohort, perf, 'exp_need_{model}', alpha=alpha, beta=beta,
                replacement_seed=17, min_selected_per_group=2)
            self.assertEqual(by_model(expected), by_model(actual))
        default = a.calculate_least_discriminatory_allocation_exp_needs(0, 0, cohort, perf, 'exp_need_{model}')
        explicit = a.calculate_least_discriminatory_allocation_exp_needs(
            0, 0, cohort, perf, 'exp_need_{model}', min_selected_per_group=1)
        self.assertEqual(default, explicit)
        pd.testing.assert_frame_equal(cohort, original)

    def test_cohort_pairing_only_exp_and_diagnostics_reset(self):
        cohort, perf = fixture()
        expanded = pd.concat([cohort.assign(person_id=cohort.person_id + 8 * i) for i in range(4)], ignore_index=True)
        df = pd.concat([expanded, perf.assign(seed=0)], ignore_index=True)
        a = allocator(df)
        identity = None
        with patch.object(a, 'calculate_topk_allocations', side_effect=AssertionError('OPT ran')):
            for minimum in (1, 2):
                opt, tru, exp, people = a.get_allocations('exp_need_{model}', algorithms='LDA-exp', min_selected_per_group=minimum)
                self.assertTrue(opt.empty and tru.empty)
                self.assertFalse(exp.empty)
                self.assertEqual(len(a.group_count_diagnostics), 4)
                self.assertTrue(all(r['min_selected_per_group'] == minimum for r in a.group_count_diagnostics))
                if identity is None:
                    identity = people[['seed', 'iteration', 'people']]
                else:
                    pd.testing.assert_frame_equal(identity, people[['seed', 'iteration', 'people']])
        a.get_allocations('exp_need_{model}', algorithms='OPT')
        self.assertEqual(a.group_count_diagnostics, [])

    def test_invalid_minimum(self):
        cohort, perf = fixture()
        a = allocator(cohort)
        for minimum in (0, -1, True, np.bool_(False), 1.0, 2.5, '2', np.nan, np.inf, 3):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                a.get_allocations('exp_need_{model}', algorithms='LDA-exp', min_selected_per_group=minimum)
        with self.assertRaisesRegex(ValueError, 'require LDA-exp'):
            a.get_allocations('exp_need_{model}', algorithms='OPT', min_selected_per_group=2)
        a.calculate_least_discriminatory_allocation_exp_needs(
            0, 0, cohort, perf, 'exp_need_{model}', min_selected_per_group=np.int64(2))


if __name__ == '__main__':
    unittest.main()
