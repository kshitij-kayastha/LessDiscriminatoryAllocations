"""Least-discriminatory allocation as a mixed-integer linear program.

The DP in allocations_parallel.py needs integer needs and costs about k^2 per
cohort. This module solves the same problem with a MILP solver, which also accepts
fractional needs. Two backends: HiGHS through scipy.optimize.milp (no licence)
and Gurobi through gurobipy (solver="gurobi"; the pip build carries a licence
limited to 2000 variables, i.e. cohorts up to n = 1999, an academic licence
lifts that).

Formulation (LLDAV2 Section 5.6). For a fixed group split t (t from group 0,
k - t from group 1), with binary z_i and a free variable d:

    minimise  d
    subject to  sum_{i in G0} z_i = t,  sum_{i in G1} z_i = k - t
                sum_i p_i z_i >= utility_requirement
                d >= m1(z) - m0(z),  d >= m0(z) - m1(z)
    where m0(z) = (1/t) sum_{G0} mu_i z_i,  m1(z) = (1/(k-t)) sum_{G1} mu_i z_i.

With t fixed the means are linear, so each split is an exact MILP. The splits
are enumerated like the DP's pairing loop and the best one wins. Ties on the
disparity go to the higher utility, the DP's rule, in one of two ways:

  tie_break="weighted"  minimise d - lam * utility in the same solve, with lam
                        small enough that no utility gain can outweigh a real
                        disparity difference (needs the smallest possible gap
                        between two distinct disparities, min_gap; for integer
                        needs that is 4 / k^2). One solve per split.
  tie_break="exact"     minimise d first, then maximise utility among the
                        allocations that reach the minimum. Two solves per
                        split that ties. The default: the weighted objective
                        turned out far slower to prove optimal (hours per
                        cohort at n = 250, k = 25).

Proving optimality is the expensive part: the LP relaxation of every split
has d = 0, so the solver must branch until it has shown no better integer
allocation exists. In practice pass time_limit (seconds per split solve) and
take the incumbent; scipy status 1 then marks a split that hit the limit.

    from src.allocations_milp import solve_milp
    selected, info = solve_milp(groups, probabilities, needs, k, utility_requirement)
"""

from contextlib import contextmanager
import os
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

# HiGHS stops at a relative gap of 1e-4 by default; we want the exact optimum.
_OPTIONS = {"disp": False, "mip_rel_gap": 0.0}
# Two disparities closer than this (in need units) count as equal when breaking
# ties, capped at 1/k^2 so integer-need optima (multiples of 1/(t(k-t)) >= 4/k^2)
# are never merged. 1e-9 made the utility stage on fractional needs fail: HiGHS
# cannot hold an exact-balance constraint, so it returned floor-utility picks.
_TIE_TOL = 1e-4


@contextmanager
def _quiet():
    """HiGHS prints a debug line from C++ on every improving solution; silence fd 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class _Failed:
    """Stand-in result when every HiGHS attempt threw."""
    x, status = None, -1


class _Result:
    def __init__(self, x, status):
        self.x, self.status = x, status


_GUROBI_ENV = None


def _gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        import gurobipy as gp
        _GUROBI_ENV = gp.Env(params={"OutputFlag": 0})
    return _GUROBI_ENV


def _run_gurobi(c, constraints, integrality, bounds, options):
    """Same problem through gurobipy. Status follows scipy's milp:
    0 optimal, 1 time limit reached (incumbent returned), 2 infeasible."""
    import gurobipy as gp
    A = constraints.A.toarray() if hasattr(constraints.A, "toarray") else np.asarray(constraints.A)
    n_var = A.shape[1]
    m = gp.Model(env=_gurobi_env())
    m.Params.Threads = 1
    m.Params.MIPGap = 0.0
    m.Params.MIPGapAbs = options.get("mip_abs_gap", 1e-10)
    if "time_limit" in options:
        m.Params.TimeLimit = options["time_limit"]
    if "random_seed" in options:
        m.Params.Seed = options["random_seed"]
    vtype = np.where(np.asarray(integrality) > 0, gp.GRB.BINARY, gp.GRB.CONTINUOUS)
    x = m.addMVar(n_var, lb=bounds.lb, ub=bounds.ub, vtype=vtype)
    m.setObjective(np.asarray(c) @ x, gp.GRB.MINIMIZE)
    lb, ub = np.asarray(constraints.lb, float), np.asarray(constraints.ub, float)
    for i in range(A.shape[0]):
        row = A[i] @ x
        if lb[i] == ub[i]:
            m.addConstr(row == lb[i])
        else:
            if np.isfinite(lb[i]):
                m.addConstr(row >= lb[i])
            if np.isfinite(ub[i]):
                m.addConstr(row <= ub[i])
    m.optimize()
    if m.SolCount == 0:
        return _Result(None, 2 if m.Status == gp.GRB.INFEASIBLE else -1)
    status = 0 if m.Status == gp.GRB.OPTIMAL else 1
    return _Result(np.array(x.X), status)


def _run(c, constraints, integrality, bounds, options, solver="highs"):
    """milp with retries: this HiGHS build intermittently throws ValueError
    ('vector') mid-solve, mostly when a time limit interrupts it. Retry with
    presolve off and other random seeds; give the split up after that."""
    if solver == "gurobi":
        return _run_gurobi(c, constraints, integrality, bounds, options)
    attempts = [options, {**options, "presolve": False},
                {**options, "random_seed": 1}, {**options, "random_seed": 2, "presolve": False}]
    for opts in attempts:
        try:
            with _quiet():
                return milp(c, constraints=constraints, integrality=integrality,
                            bounds=bounds, options=opts)
        except ValueError:
            continue
    return _Failed()


def _split_milp(groups, probabilities, needs, k, t, utility_requirement, *,
                utility_weight=0.0, max_disparity=None, time_limit=None, solver="highs"):
    """Solve one split. Returns (status, selected, disparity, utility).

    utility_weight > 0 minimises d - weight * utility. With max_disparity set
    the objective is maximum utility subject to disparity <= max_disparity."""
    n = len(probabilities)
    g0, g1 = groups
    weights = np.zeros(n)
    weights[g0] = -needs[g0] / t             # -m0(z)
    weights[g1] = needs[g1] / (k - t)        # +m1(z)
    count_0 = np.zeros(n); count_0[g0] = 1
    count_1 = np.zeros(n); count_1[g1] = 1

    # variables: z_0..z_{n-1}, d
    rows, lb, ub = [], [], []
    rows.append(np.append(count_0, 0.0)); lb.append(t); ub.append(t)
    rows.append(np.append(count_1, 0.0)); lb.append(k - t); ub.append(k - t)
    rows.append(np.append(probabilities, 0.0)); lb.append(utility_requirement); ub.append(np.inf)
    rows.append(np.append(-weights, 1.0)); lb.append(0.0); ub.append(np.inf)   # d >= m1 - m0
    rows.append(np.append(weights, 1.0)); lb.append(0.0); ub.append(np.inf)    # d >= m0 - m1
    constraints = LinearConstraint(csr_matrix(np.vstack(rows)), lb, ub)
    d_upper = np.inf if max_disparity is None else max_disparity
    bounds = Bounds(np.append(np.zeros(n), 0.0), np.append(np.ones(n), d_upper))
    integrality = np.append(np.ones(n), 0)
    if max_disparity is None:
        c = np.append(-utility_weight * probabilities, 1.0)
    else:
        c = np.append(-probabilities, 0.0)
    options = dict(_OPTIONS)
    if utility_weight:
        options["mip_abs_gap"] = 0.0
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = _run(c, constraints, integrality, bounds, options, solver)
    if res.x is None:
        return res.status, None, np.inf, 0.0
    z = res.x[:n] > 0.5
    selected = np.flatnonzero(z)
    m0 = needs[g0][z[g0]].mean()
    m1 = needs[g1][z[g1]].mean()
    return res.status, selected.tolist(), abs(m1 - m0), float(probabilities[z].sum())


def solve_milp(groups, probabilities, needs, k, utility_requirement, *,
               tie_break="exact", min_gap=None, time_limit=None, tie_tol=None,
               solver="highs"):
    """Least-discriminatory allocation of k people under the utility floor.

    groups: two lists of row indices (group 0, group 1); probabilities and
    needs: arrays over the cohort; needs may be fractional. min_gap is the
    smallest difference between two distinct achievable disparities (default
    4 / k^2 for integer needs, 0 for fractional ones, where ties do not occur
    in practice). Returns (selected indices, info) with the disparity, utility,
    split count, solve time and the scipy status of the winning split;
    selected is None when no split is feasible (the DP's fallback case).
    tie_tol: disparities within this of the minimum count as tied (default
    min(1e-4, 1/k^2)). solver: "highs" (scipy) or "gurobi" (gurobipy).
    """
    if solver not in ("highs", "gurobi"):
        raise ValueError("solver must be 'highs' or 'gurobi'")
    if tie_tol is None:
        tie_tol = min(_TIE_TOL, 1.0 / k**2)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    needs = np.asarray(needs, dtype=np.float64)
    g0, g1 = (np.asarray(g, dtype=np.int64) for g in groups)
    if tie_break not in ("weighted", "exact", None):
        raise ValueError("tie_break must be 'weighted', 'exact' or None")
    weight = 0.0
    if tie_break == "weighted":
        if min_gap is None:
            min_gap = 4.0 / k**2 if np.all(needs == np.rint(needs)) else 0.0
        if min_gap > 0:
            # utility can exceed the floor by at most (top-k sum - floor)
            top = np.sort(probabilities)[-k:].sum()
            room = max(top - utility_requirement, 1e-12)
            weight = 0.5 * min_gap / room
        else:
            weight = 1e-9
    started = perf_counter()
    results = []
    for t in range(max(1, k - len(g1)), min(len(g0), k - 1) + 1):
        status, selected, disparity, utility = _split_milp(
            (g0, g1), probabilities, needs, k, t, utility_requirement,
            utility_weight=weight, time_limit=time_limit, solver=solver)
        if selected is not None:
            results.append((disparity, utility, t, selected, status))
    info = {"splits": len(results), "seconds": perf_counter() - started,
            "tie_break": tie_break, "solver": solver}
    if not results:
        return None, {**info, "disparity": np.inf, "utility": 0.0, "status": None}
    best = min(r[0] for r in results)
    tied = [r for r in results if r[0] <= best + tie_tol]
    if tie_break == "exact":
        improved = []
        for disparity, utility, t, selected, status in tied:
            status2, selected2, disparity2, utility2 = _split_milp(
                (g0, g1), probabilities, needs, k, t, utility_requirement,
                max_disparity=best + tie_tol, time_limit=time_limit, solver=solver)
            improved.append((disparity2, utility2, t, selected2, status2) if selected2 is not None
                            else (disparity, utility, t, selected, status))
        tied = improved
    disparity, utility, t, selected, status = max(tied, key=lambda r: r[1])
    info.update({"disparity": float(disparity), "utility": utility, "split": t,
                 "status": status, "seconds": perf_counter() - started})
    return sorted(selected), info
