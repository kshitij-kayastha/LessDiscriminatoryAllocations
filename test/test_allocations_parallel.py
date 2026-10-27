"""Run: python -B -m unittest discover -s test -p 'test_allocations_parallel.py'."""

import itertools
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.allocations import Allocations as ReferenceAllocations
from src.allocations_parallel import Allocations, ParallelAllocations, _need_values


def fixture(size=10, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'person_id': np.arange(size) * 3 + 10, 'seed': 0,
        'race': [0] * (size // 2) + [1] * (size - size // 2),
        'h': rng.integers(0, 8, size), 'y': rng.integers(0, 2, size),
    })
    for m in range(1, 5):
        frame[f'm_{m}'] = rng.uniform(0.01, .99, size).astype(dtype)
        frame[f'exp_need_m_{m}'] = rng.uniform(0, 8, size)
    perf = pd.DataFrame({'person_id': [-1], 'seed': [0],
                         'm_1': [.1], 'm_2': [.105], 'm_3': [.4], 'm_4': [.109]})
    return frame, perf


def allocator(cls, df, *, rate=.4, tolerance=.1, **kwargs):
    if cls is Allocations:
        kwargs.setdefault('show_progress', False)
        kwargs.setdefault('n_jobs', 1)
    return cls(df, 10, rate, 'y', 'h', 'race', .01, tolerance,
               n_models=4, iters_per_seed=2, **kwargs)


def methods(a, frame, perf):
    return (
        a.calculate_topk_allocations(0, 0, frame, perf),
        a.calculate_least_discriminatory_allocation(0, 0, frame, perf),
        a.calculate_least_discriminatory_allocation_exp_needs(0, 0, frame, perf, 'exp_need_{model}'),
    )


def by_model(rows):
    return {m: r['selected'] for r in rows for m in r['models']}


class ParallelAllocationTests(unittest.TestCase):
    def test_reference_equivalence_random_and_float32(self):
        for seed, dtype, rate, tolerance in itertools.product(
                range(4), (np.float32, np.float64), (.2, .5, 1), (0, .1)):
            with self.subTest(seed=seed, dtype=dtype, rate=rate, tolerance=tolerance):
                frame, perf = fixture(seed=seed, dtype=dtype)
                frame.index = np.arange(len(frame)) * 11 + 9
                before = frame.copy(deep=True)
                old = allocator(ReferenceAllocations, frame, rate=rate, tolerance=tolerance)
                new = allocator(Allocations, frame, rate=rate, tolerance=tolerance)
                self.assertEqual(methods(new, frame, perf), methods(old, frame, perf))
                pd.testing.assert_frame_equal(frame, before)

    def test_exact_ties_deduplication_and_block_boundaries(self):
        frame, perf = fixture()
        for m in range(1, 5):
            frame[f'm_{m}'] = .5
            frame[f'exp_need_m_{m}'] = frame.h
        old = methods(allocator(ReferenceAllocations, frame), frame, perf)
        with patch('src.allocations_parallel._PAIR_BLOCK_ENTRIES', 3):
            new = methods(allocator(Allocations, frame), frame, perf)
        self.assertEqual(new, old)
        for rows in new:
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['models'], ['m_1', 'm_2', 'm_4'])

    def test_exhaustive_disparity_and_utility(self):
        frame, perf = fixture()
        # Binary fractions make independent utility checks free of sum-order noise.
        for m in range(1, 5):
            frame[f'm_{m}'] = np.rint(frame[f'm_{m}'] * 32) / 32
        a = allocator(Allocations, frame, tolerance=.125)
        for rounded, result in zip((False, True), methods(a, frame, perf)[1:]):
            for model, identities in by_model(result).items():
                needs = np.rint(frame[f'exp_need_{model}']) if rounded else frame.h
                floor = frame[model].nlargest(a.k).sum() - a.k * a.allocation_eps

                def objective(indices):
                    groups = [[i for i in indices if frame.loc[i, 'race'] == g] for g in (0, 1)]
                    utility = frame.loc[list(indices), model].sum()
                    if not all(groups) or utility < floor:
                        return None
                    disparity = abs(sum(needs.loc[groups[1]]) / len(groups[1])
                                    - sum(needs.loc[groups[0]]) / len(groups[0]))
                    return disparity, -utility

                choices = [score for indices in itertools.combinations(frame.index, a.k)
                           if (score := objective(indices)) is not None]
                chosen = frame.index[frame.person_id.isin(identities)].tolist()
                if choices:
                    self.assertEqual(objective(chosen), min(choices))
                else:
                    self.assertEqual(set(chosen), set(frame[model].nlargest(a.k).index))

    def test_true_needs_and_rounding(self):
        values = pd.Series([0, .5, 1.5, 2.5, 3.5, 4.51], name='needs')
        self.assertEqual(_need_values(values, rounded=True, k=4), [0, 0, 2, 2, 4, 5])
        self.assertEqual(_need_values(pd.Series([0., 1., 4.]), rounded=False, k=4), [0, 1, 4])
        for bad in ([1.2], [-1], [np.nan], [np.inf], [2**53]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _need_values(pd.Series(bad), rounded=False, k=4)

    def test_topk_fallback_and_one_person_group(self):
        frame, perf = fixture()
        frame.race = [0] * 9 + [1]
        for m in range(1, 5):
            frame[f'm_{m}'] = [.9] * 9 + [.01]
        old = allocator(ReferenceAllocations, frame, tolerance=0)
        new = allocator(Allocations, frame, tolerance=0)
        self.assertEqual(methods(new, frame, perf), methods(old, frame, perf))
        self.assertTrue(all(row['fallback_used'] for row in new.allocation_diagnostics
                            if row['algorithm'] != 'OPT'))

    def test_only_requested_inputs_are_used(self):
        frame, perf = fixture()
        a = allocator(Allocations, frame)
        invalid = frame.drop(columns=['h'] + [f'exp_need_m_{m}' for m in range(1, 5)])
        with patch('src.allocations_parallel._solve_dp', side_effect=AssertionError('DP ran')):
            a.calculate_topk_allocations(0, 0, invalid, perf)
        # Predicted-needs DP need not read any true-needs column.
        expected = a.calculate_least_discriminatory_allocation_exp_needs(0, 0, frame, perf)
        actual = a.calculate_least_discriminatory_allocation_exp_needs(0, 0, frame.drop(columns='h'), perf)
        self.assertEqual(actual, expected)
        # Excluded model m_3 may have invalid predictions; it must never run.
        frame['m_3'] = np.nan
        frame['exp_need_m_3'] = np.nan
        self.assertEqual(a.calculate_least_discriminatory_allocation_exp_needs(0, 0, frame, perf), expected)

    def test_shared_cohorts_and_end_to_end_processes(self):
        frame, perf = fixture(size=60)
        frames = []
        for seed in (0, 1):
            frames.append(frame.assign(seed=seed))
            frames.append(perf.assign(seed=seed))
        df = pd.concat(frames, ignore_index=True)
        old = allocator(ReferenceAllocations, df).get_allocations('exp_need_{model}')
        sequential = allocator(Allocations, df)
        serial = sequential.get_allocations('exp_need_{model}')
        parallel = allocator(Allocations, df, n_jobs=2)
        actual = parallel.get_allocations('exp_need_{model}')
        for index in range(3):
            pd.testing.assert_frame_equal(actual[index], serial[index])
            pd.testing.assert_frame_equal(actual[index], old[index])
        pd.testing.assert_frame_equal(actual[3], old[3][['seed', 'iteration', 'people']])
        self.assertEqual(len(parallel.allocation_diagnostics), 4 * 3 * 3)
        self.assertTrue(all(row['worker_pid'] != os.getpid() for row in parallel.allocation_diagnostics))
        exp_only = parallel.get_allocations('exp_need_{model}', algorithms='LDA-exp')
        self.assertTrue(exp_only[0].empty and exp_only[1].empty)
        pd.testing.assert_frame_equal(exp_only[2], actual[2])
        pd.testing.assert_frame_equal(exp_only[3], actual[3])
        self.assertEqual(len(parallel.allocation_diagnostics), 4 * 3)

    def test_validation(self):
        frame, perf = fixture()
        for jobs in (0, True, 1.5):
            with self.subTest(n_jobs=jobs), self.assertRaises(ValueError):
                allocator(Allocations, frame, n_jobs=jobs)
        a = allocator(Allocations, frame)
        for column, value in [('race', 2), ('h', 1.5), ('m_1', np.inf), ('person_id', frame.person_id.iloc[1])]:
            broken = frame.copy()
            broken[column] = broken[column].astype(float)
            broken.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                a.calculate_least_discriminatory_allocation(0, 0, broken, perf)
        for algorithms in ([], ['unknown'], 'LDA-exp'):
            with self.subTest(algorithms=algorithms), self.assertRaises(ValueError):
                a.get_allocations(algorithms=algorithms)
        self.assertIs(ParallelAllocations, Allocations)


if __name__ == '__main__':
    unittest.main()
