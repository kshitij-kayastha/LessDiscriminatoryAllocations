"""Run: python -B -m unittest discover -s test -p 'test_allocations_milp.py'.

The MILP must reach the DP's disparity and utility on integer needs (the
selected people may differ only on exact ties), and both must agree with a
brute-force search over every feasible allocation on small cohorts.
"""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.allocations_parallel import _solve_dp  # noqa: E402
from src.allocations_milp import solve_milp  # noqa: E402


def random_cohort(rng, n, integer_needs=True):
    group = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(group)
    groups = [np.flatnonzero(group == g).tolist() for g in (0, 1)]
    probabilities = rng.uniform(0.01, 0.99, n)
    needs = rng.integers(0, 8, n).astype(float) if integer_needs else rng.uniform(0, 8, n)
    return groups, probabilities, needs


def floor_and_topk(probabilities, k, eps):
    top_k = np.argsort(-probabilities, kind="stable")[:k].tolist()
    return probabilities[top_k].sum() - k * eps, top_k


def evaluate(selected, groups, probabilities, needs):
    sel = np.zeros(len(probabilities), bool)
    sel[selected] = True
    m = [needs[g][sel[g]].mean() for g in groups]
    return abs(m[1] - m[0]), probabilities[sel].sum()


def brute_force(groups, probabilities, needs, k, floor):
    best = (np.inf, -np.inf)
    for combo in itertools.combinations(range(len(probabilities)), k):
        sel = list(combo)
        if not all(any(i in g for i in sel) for g in groups):
            continue
        d, u = evaluate(sel, groups, probabilities, needs)
        if u < floor - 1e-12:
            continue
        if d < best[0] - 1e-12 or (abs(d - best[0]) <= 1e-12 and u > best[1]):
            best = (d, u)
    return best


class MilpTests(unittest.TestCase):
    def test_matches_dp_and_brute_force_on_integer_needs(self):
        rng = np.random.default_rng(0)
        for trial in range(40):
            n = int(rng.integers(6, 13))
            k = int(rng.integers(2, n))
            eps = float(rng.choice([0.0, 0.1, 0.3]))
            groups, p, needs = random_cohort(rng, n)
            floor, top_k = floor_and_topk(p, k, eps)
            with self.subTest(trial=trial, n=n, k=k, eps=eps):
                dp_sel, fallback = _solve_dp(groups, p.tolist(), [int(v) for v in needs],
                                             k, floor, top_k)
                milp_sel, info = solve_milp(groups, p, needs, k, floor)
                if fallback:
                    # At eps = 0 the floor equals the top-k utility and the DP
                    # can reject the top-k set itself by ~1e-16 of summation
                    # order; the MILP then returns that set within tolerance.
                    if milp_sel is not None:
                        self.assertGreaterEqual(info["utility"], floor - 1e-9)
                    continue
                d_dp, u_dp = evaluate(dp_sel, groups, p, needs)
                d_bf, u_bf = brute_force(groups, p, needs, k, floor)
                self.assertAlmostEqual(d_dp, d_bf, places=9)
                self.assertAlmostEqual(u_dp, u_bf, places=9)
                self.assertAlmostEqual(info["disparity"], d_dp, places=7)
                self.assertAlmostEqual(info["utility"], u_dp, places=7)

    def test_fractional_needs_match_brute_force(self):
        rng = np.random.default_rng(1)
        for trial in range(20):
            n = int(rng.integers(6, 12))
            k = int(rng.integers(2, n))
            groups, p, needs = random_cohort(rng, n, integer_needs=False)
            floor, _ = floor_and_topk(p, k, 0.1)
            with self.subTest(trial=trial, n=n, k=k):
                milp_sel, info = solve_milp(groups, p, needs, k, floor)
                d_bf, u_bf = brute_force(groups, p, needs, k, floor)
                self.assertIsNotNone(milp_sel)
                self.assertAlmostEqual(info["disparity"], d_bf, places=7)
                self.assertAlmostEqual(info["utility"], u_bf, places=7)


if __name__ == "__main__":
    unittest.main()
