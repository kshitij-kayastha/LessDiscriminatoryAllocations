"""Parallel baseline allocations for integer true / rounded predicted needs.

Notebook usage (keep the existing constructor arguments)::

    from src.allocations_parallel import Allocations

    a = Allocations(
        df, sample_size=1000, selection_rate=0.1,
        qualification_col="y", need_col="gagne_sum_t", group_col="race",
        rashomon_eps=0.01, allocation_eps=0.1,
        n_models=100, iters_per_seed=5, n_jobs=4,
    )
    opt, tru, exp, people = a.get_allocations(
        exp_need_col="exp_need_{model}",
        algorithms=["OPT", "LDA-tru", "LDA-exp"],
    )

Use algorithms="LDA-exp" to compute only that allocation. n_jobs=1 runs the
same implementation sequentially; -1 uses all available CPUs. Four workers
is a conservative starting point: each active DP needs its own group tables.
Joblib's loky processes work from an imported module in Jupyter on macOS;
no notebook-defined worker functions or multiprocessing main guard is needed.

The default cohort sampling, validation-loss Rashomon rule, utility floor,
one-person-per-group DP constraint, top-k fallback, and allocation CSV columns
match allocations.py. Each seed/cohort is sampled once in the parent process.
Each worker evaluates one model, and results are merged in model order, so
worker completion order cannot change allocations or allocation_idx values.

True needs must be finite nonnegative integers. Predicted needs are rounded
once with np.rint (halfway values to even: 2.5 -> 2, 3.5 -> 4). This module has
no replacement, error-shrinkage, fractional-grid, or extra group-count options.

The DP uses dictionaries with the original insertion/tie order, numeric
arrays instead of pandas in its inner loops, integer bit masks for selected
subsets, and bounded NumPy blocks for comparing the two groups' states.
The four returned DataFrames are (OPT, LDA-tru, LDA-exp, people). The people
table contains seed, iteration, people. allocation_diagnostics additionally
records per-model DP fallback, selected group counts, and elapsed time.
"""

from contextlib import contextmanager
from numbers import Integral, Real
import os
from time import perf_counter

from joblib import Parallel, delayed, effective_n_jobs, parallel_config
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


_ALGORITHMS = ("OPT", "LDA-tru", "LDA-exp")
_ALLOCATION_COLUMNS = ["seed", "iteration", "allocation_idx", "selected", "k'", "n'", "models"]
_PAIR_BLOCK_ENTRIES = 262_144


def _group_tables(indices, probabilities, needs, max_count):
    """Maximize utility for each (count, integer need sum), keeping first ties."""
    tables = [{} for _ in range(max_count + 1)]
    tables[0][0] = (0.0, 0)
    for processed, i in enumerate(indices, start=1):
        need, probability, bit = needs[i], probabilities[i], 1 << i
        for count in range(min(processed, max_count), 0, -1):
            current = tables[count]
            for previous_sum, (previous_utility, previous_mask) in tables[count - 1].items():
                need_sum = previous_sum + need
                utility = previous_utility + probability
                incumbent = current.get(need_sum)
                if incumbent is None or utility > incumbent[0]:
                    current[need_sum] = (utility, previous_mask | bit)
    return tables


def _best_mask(tables, group_sizes, k, utility_requirement):
    """Compare all feasible pairs with bounded memory and legacy tie-breaking.

    Counts and sum states retain their original traversal order. The first
    pair wins an exact tie in both floating-point disparity and utility.
    Do not replace the difference of means by an algebraically equivalent
    fraction: its floating-point ties can differ from the existing DP.
    """
    best_disparity, best_utility, best_mask = np.inf, 0.0, None
    for count_0 in range(max(1, k - group_sizes[1]), min(group_sizes[0], k - 1) + 1):
        count_1 = k - count_0
        table_0, table_1 = tables[0][count_0], tables[1][count_1]
        if not table_0 or not table_1:
            continue
        states_0, states_1 = list(table_0.values()), list(table_1.values())
        means_0 = np.asarray(list(table_0), dtype=np.float64) / count_0
        means_1 = np.asarray(list(table_1), dtype=np.float64) / count_1
        utilities_0 = np.asarray([state[0] for state in states_0])
        utilities_1 = np.asarray([state[0] for state in states_1])
        rows_per_block = max(1, _PAIR_BLOCK_ENTRIES // len(states_1))
        for start in range(0, len(states_0), rows_per_block):
            stop = min(start + rows_per_block, len(states_0))
            utilities = utilities_0[start:stop, None] + utilities_1[None, :]
            disparities = np.abs(means_1[None, :] - means_0[start:stop, None])
            disparities[utilities < utility_requirement] = np.inf
            disparity = disparities.min()
            if not np.isfinite(disparity) or disparity > best_disparity:
                continue
            # argmax returns the first row-major pair in an exact utility tie.
            utilities[disparities != disparity] = -np.inf
            flat_index = int(utilities.argmax())
            row, column = divmod(flat_index, len(states_1))
            utility = utilities[row, column]
            if disparity < best_disparity or utility > best_utility:
                best_disparity, best_utility = disparity, utility
                best_mask = states_0[start + row][1] | states_1[column][1]
    return best_mask


def _solve_dp(groups, probabilities, needs, k, utility_requirement, top_k):
    tables = [_group_tables(group, probabilities, needs, min(len(group), k - 1))
              for group in groups]
    mask = _best_mask(tables, [len(group) for group in groups], k, utility_requirement)
    if mask is None:
        return list(top_k), True
    selected = []
    while mask:
        bit = mask & -mask
        selected.append(bit.bit_length() - 1)
        mask ^= bit
    return selected, False


def _model_task(model, algorithms, groups, probabilities, true_needs,
                expected_needs, k, utility_requirement, top_k):
    """Module-level, pickleable worker; does not read or modify shared state."""
    selections, diagnostics = {}, []
    # Python floats are faster in the state loop and preserve float64 arithmetic.
    # Keep float32 scalars if supplied: the original DP accumulates in float32.
    values = probabilities.tolist() if probabilities.dtype == np.float64 else probabilities
    for algorithm in algorithms:
        started = perf_counter()
        if algorithm == "OPT":
            selected, fallback = list(top_k), False
        else:
            needs = true_needs if algorithm == "LDA-tru" else expected_needs
            selected, fallback = _solve_dp(groups, values, needs, k, utility_requirement, top_k)
        selections[algorithm] = selected
        selected_set = set(selected)
        diagnostics.append({
            "model": model, "algorithm": algorithm, "fallback_used": fallback,
            "selected_count_0": sum(i in selected_set for i in groups[0]),
            "selected_count_1": sum(i in selected_set for i in groups[1]),
            "utility_requirement": utility_requirement,
            "elapsed_seconds": perf_counter() - started, "worker_pid": os.getpid(),
        })
    return model, selections, diagnostics


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _need_values(series, *, rounded, k):
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{series.name} must contain finite nonnegative needs")
    if rounded:
        values = np.rint(values)
    elif (values != np.floor(values)).any():
        raise ValueError(f"{series.name} must contain integer true needs")
    if (values > (2**53 - 1) // k).any():
        raise ValueError(f"{series.name} is too large for exact integer need sums")
    return [int(value) for value in values]


class Allocations:
    """Baseline allocation interface with process parallelism across models.

    n_jobs controls concurrent models, not cohorts or group-table threads.
    n_jobs=1 is useful for debugging; n_jobs=-1 uses every available CPU.
    Cohort identities and merged outputs are independent of n_jobs.
    """

    def __init__(self, df, sample_size, selection_rate, qualification_col, need_col,
                 group_col, rashomon_eps, allocation_eps, n_models=None,
                 iters_per_seed=5, *, n_jobs=4, show_progress=True):
        self.df = df
        self.sample_size = _positive_int("sample_size", sample_size)
        if (isinstance(selection_rate, (bool, np.bool_)) or not isinstance(selection_rate, Real)
                or not np.isfinite(selection_rate) or not 0 < selection_rate <= 1):
            raise ValueError("selection_rate must be finite and in (0, 1]")
        self.selection_rate = selection_rate
        self.k = int(selection_rate * sample_size)
        if self.k < 1:
            raise ValueError("selection_rate * sample_size must select at least one person")
        self.qualification_col, self.need_col, self.group_col = qualification_col, need_col, group_col
        for name, value in (("rashomon_eps", rashomon_eps), ("allocation_eps", allocation_eps)):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.rashomon_eps, self.allocation_eps = rashomon_eps, allocation_eps
        if n_models is None:
            n_models = sum(isinstance(col, str) and col.startswith("m_") for col in df.columns)
        self.n_models = _positive_int("n_models", n_models)
        self.iters_per_seed = _positive_int("iters_per_seed", iters_per_seed)
        if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, Integral) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer (use -1 for all CPUs)")
        self.n_jobs, self.show_progress = int(n_jobs), bool(show_progress)
        self.n_seeds = df["seed"].nunique() if "seed" in df else 0
        self.allocation_diagnostics = []

    def rashomon_models(self, df_perf):
        names = [f"m_{i}" for i in range(1, self.n_models + 1)]
        rows = df_perf.loc[df_perf["person_id"] == -1, names]
        if len(rows) != 1:
            raise ValueError("Expected exactly one validation-loss row per split")
        losses = rows.iloc[0].astype(float)
        if not np.isfinite(losses).all():
            raise ValueError("Validation losses must be finite")
        return losses.index[losses <= losses.min() + self.rashomon_eps].tolist()

    @staticmethod
    def exp_need_col_for_model(model_name, exp_need_col="exp_need_{model}"):
        if not isinstance(exp_need_col, str) or not exp_need_col:
            raise ValueError("exp_need_col must be a column name or template")
        try:
            return exp_need_col.format(model=model_name)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("Use {model} as the placeholder, e.g. exp_need_{model}") from error

    @contextmanager
    def _executor(self):
        workers = min(self.n_models, effective_n_jobs(self.n_jobs))
        # Avoid each process starting its own additional NumPy/BLAS thread pool.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with Parallel(n_jobs=workers, return_as="generator_unordered",
                          batch_size=1, pre_dispatch="n_jobs") as executor:
                yield executor

    def _evaluate_cohort(self, seed, iteration, df, perf, algorithms, exp_need_col, executor):
        if not 1 <= self.k <= len(df):
            raise ValueError("Selection count must be between one and the cohort size")
        if not df["person_id"].is_unique or df["person_id"].isna().any():
            raise ValueError("A cohort must contain unique, nonmissing person IDs")
        if not df[self.group_col].isin([0, 1]).all():
            raise ValueError("Group labels must be 0 or 1")
        if not df[self.qualification_col].isin([0, 1]).all():
            raise ValueError("Qualification values must be 0 or 1")
        group_values = df[self.group_col].to_numpy()
        groups = [np.flatnonzero(group_values == group).tolist() for group in (0, 1)]
        if any(algorithm != "OPT" for algorithm in algorithms):
            if self.k < 2 or not all(groups):
                raise ValueError("The DP needs k >= 2 and at least one person in each group")
        models = self.rashomon_models(perf)
        true_needs = (_need_values(df[self.need_col], rounded=False, k=self.k)
                      if "LDA-tru" in algorithms else None)
        tasks = []
        for model in models:
            probabilities = df[model].to_numpy(copy=True)
            if probabilities.dtype not in (np.dtype("float32"), np.dtype("float64")):
                probabilities = probabilities.astype(np.float64)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError(f"{model} must contain finite probabilities in [0, 1]")
            # pandas preserves the legacy top-k order, including tied values.
            series = pd.Series(probabilities)
            top_k = series.nlargest(self.k).index.to_list()
            utility_requirement = series.loc[top_k].sum().item() - self.k * self.allocation_eps
            expected_needs = None
            if "LDA-exp" in algorithms:
                column = self.exp_need_col_for_model(model, exp_need_col)
                expected_needs = _need_values(df[column], rounded=True, k=self.k)
            tasks.append((model, algorithms, groups, probabilities, true_needs,
                          expected_needs, self.k, utility_requirement, top_k))
        # Top-k alone is cheap; avoid process startup for OPT-only requests.
        completed = ((_model_task(*args) for args in tasks) if algorithms == ("OPT",)
                     else executor(delayed(_model_task)(*args) for args in tasks))
        results = {}
        for model, selections, diagnostics in tqdm(
                completed, total=len(models), disable=not self.show_progress,
                desc=f"Seed {seed}, cohort {iteration}", unit="model"):
            results[model] = (selections, diagnostics)
        person_ids = df["person_id"].to_numpy()
        qualifications = df[self.qualification_col].to_numpy()
        qualified_count = int(qualifications.sum())
        records = {algorithm: [] for algorithm in algorithms}
        allocation_ids = {algorithm: {} for algorithm in algorithms}
        for model in models:
            selections, diagnostics = results[model]
            self.allocation_diagnostics.extend(
                {"seed": seed, "iteration": iteration, **row} for row in diagnostics)
            for algorithm in algorithms:
                selected = selections[algorithm]
                identities = tuple(sorted(person_ids[selected].tolist()))
                indices = allocation_ids[algorithm]
                if identities not in indices:
                    index = len(records[algorithm])
                    indices[identities] = index
                    records[algorithm].append({
                        "seed": seed, "iteration": iteration, "allocation_idx": index,
                        "selected": list(identities), "k'": int(qualifications[selected].sum()),
                        "n'": qualified_count, "models": [model],
                    })
                else:
                    records[algorithm][indices[identities]]["models"].append(model)
        return records

    def calculate_topk_allocations(self, seed, iteration, df, perf):
        with self._executor() as executor:
            return self._evaluate_cohort(seed, iteration, df, perf, ("OPT",), None, executor)["OPT"]

    def calculate_least_discriminatory_allocation(self, seed, iteration, df, perf):
        """DP on integer true needs, parallel over eligible models."""
        with self._executor() as executor:
            return self._evaluate_cohort(seed, iteration, df, perf, ("LDA-tru",), None, executor)["LDA-tru"]

    def calculate_least_discriminatory_allocation_exp_needs(
            self, seed, iteration, df, perf, exp_need_col="exp_need_{model}"):
        """DP on np.rint(expected needs), parallel over eligible models."""
        with self._executor() as executor:
            return self._evaluate_cohort(seed, iteration, df, perf, ("LDA-exp",), exp_need_col, executor)["LDA-exp"]

    def get_allocations(self, exp_need_col=None, *, algorithms=None):
        """Sample shared cohorts and return OPT, true DP, predicted DP, people.

        Defaults to OPT and LDA-tru, plus LDA-exp when exp_need_col is provided.
        Pass a name or a list of names to run only the algorithms needed.
        Unrequested algorithms return empty DataFrames with the usual schema.
        The cohort sampler exactly follows the baseline RNG sequence for the
        usual contiguous seeds. No prediction values are used for sampling.
        """
        if algorithms is None:
            algorithms = ["OPT", "LDA-tru"] + (["LDA-exp"] if exp_need_col is not None else [])
        if isinstance(algorithms, str):
            algorithms = [algorithms]
        requested = set(algorithms)
        if not requested or not requested <= set(_ALGORITHMS):
            raise ValueError("algorithms must contain OPT, LDA-tru, and/or LDA-exp")
        algorithms = tuple(name for name in _ALGORITHMS if name in requested)
        if "LDA-exp" in algorithms and exp_need_col is None:
            raise ValueError("exp_need_col is required for LDA-exp")
        if self.df["seed"].isna().any():
            raise ValueError("Seed values must not be missing")
        seeds = sorted(self.df["seed"].unique())
        if not seeds:
            raise ValueError("No seed splits were supplied")
        self.allocation_diagnostics = []
        allocations = {algorithm: [] for algorithm in _ALGORITHMS}
        people = []
        rng = np.random.default_rng(0)
        with self._executor() as executor:
            for seed in seeds:
                split = self.df[self.df["seed"] == seed]
                population = split[split["person_id"] >= 0]
                perf = split[split["person_id"] < 0]
                if not population[self.group_col].isin([0, 1]).all():
                    raise ValueError("Population group labels must be 0 or 1")
                if not population["person_id"].is_unique:
                    raise ValueError("Person IDs must be unique within each seed's population")
                groups = [population[population[self.group_col] == group] for group in (0, 1)]
                if self.sample_size < 2:
                    raise ValueError("Cohort sampling needs sample_size >= 2")
                for iteration in range(self.iters_per_seed):
                    count_0 = int(rng.choice(range(1, self.sample_size)))
                    counts = [count_0, self.sample_size - count_0]
                    if any(count > len(group) for count, group in zip(counts, groups)):
                        raise ValueError(
                            f"Seed {seed}, cohort {iteration}: sampled group counts {counts} "
                            f"exceed available counts {[len(group) for group in groups]}")
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


ParallelAllocations = Allocations
