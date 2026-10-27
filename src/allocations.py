from fractions import Fraction
from numbers import Real

import numpy as np
import pandas as pd
from tqdm import tqdm


class Allocations:
    def __init__(self, df, sample_size, selection_rate, qualification_col, need_col, group_col, rashomon_eps, allocation_eps, n_models=None, iters_per_seed=5):
        self.df = df
        self.sample_size = sample_size
        self.selection_rate = selection_rate
        self.k = int(self.selection_rate*self.sample_size)
        self.qualification_col = qualification_col
        self.need_col = need_col
        self.group_col = group_col
        self.rashomon_eps = rashomon_eps
        self.allocation_eps = allocation_eps
        self.n_models = n_models if n_models else len([c for c in self.df.columns if c.startswith("m_")])
        self.n_seeds = len(self.df["seed"].unique())
        self.iters_per_seed = iters_per_seed
        self.need_diagnostics = []
        self.group_count_diagnostics = []

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
    def exp_need_col_for_model(model_name, exp_need_col="expected_need_{model}"):
        if not isinstance(exp_need_col, str) or not exp_need_col:
            raise ValueError("expected_need_column must be a column name or template")
        try:
            return exp_need_col.format(model=model_name)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("Use {model} as the placeholder, e.g. expected_need_{model}") from error

    def calculate_topk_allocations(self, seed, iteration, df, perf):
        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model in self.rashomon_models(perf):
            selected_idx = df[model].nlargest(self.k).index.to_list()
            k_prime = int(df.loc[selected_idx, self.qualification_col].sum())
            n_prime = int(df[self.qualification_col].sum())
            selected_people = df.loc[selected_idx, "person_id"].to_list()

            selected_people.sort()
            allocation = tuple(selected_people)
            if allocation not in allocations:
                allocation_data.append({
					"seed":seed,
					"iteration":iteration,
					"allocation_idx":allocation_idx,
					"selected": selected_people,
					"k'": k_prime,
					"n'": n_prime,
				})
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": [model]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append(model)

        models = {}
        for v in allocations.values():
            models[v["allocation_idx"]] = v["models"]

        for i in range(len(allocation_data)):
            allocation_data[i]["models"] = models[allocation_data[i]["allocation_idx"]] 

        return allocation_data

    def make_group_tables(self, selected_idx, probabilities, needs, max_count):
        dp = [{} for _ in range(max_count + 1)]
        dp[0][0] = (0.0, ())

        for processed, i in enumerate(selected_idx, start=1):
            for count in range(min(processed, max_count), 0,  -1):
                for need_sum, previous in dp[count - 1].items():
                    new_sum = need_sum + needs.loc[i]
                    new_utility = previous[0] + probabilities.loc[i]
                    incumbent = dp[count].get(new_sum)

                    if incumbent is None or new_utility > incumbent[0]:
                        dp[count][new_sum] = (new_utility, previous[1] + (i, ))
        return dp

    def _validate_min_selected_per_group(self, minimum):
        if (isinstance(minimum, (bool, np.bool_))
                or not isinstance(minimum, (int, np.integer)) or minimum < 1):
            raise ValueError("min_selected_per_group must be a positive integer")
        if 2 * minimum > self.k:
            raise ValueError("min_selected_per_group requires 2 * minimum <= k")

    def calculate_least_discriminatory_allocation(
        self, seed, iteration, df, perf, needs=None, *, needs_by_model=None, need_scale=1,
        min_selected_per_group=1, diagnostics=None,
    ):
        """Minimize disparity subject to utility and a selected count per group.

        If the requested minimum is infeasible, use the original minimum-one
        DP on the same needs and probabilities. Its existing top-k fallback is
        retained if even minimum one is infeasible. An optional diagnostics list
        receives one record per model, before identical allocations are merged.
        The default minimum of one preserves the original allocation behavior.
        """
        self._validate_min_selected_per_group(min_selected_per_group)
        if needs is not None and needs_by_model is not None:
            raise ValueError("Pass either shared needs or needs_by_model, not both")
        if needs is None and needs_by_model is None:
            needs = df[self.need_col]

        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model in self.rashomon_models(perf):

            probabilities = df[model]
            model_needs = needs if needs_by_model is None else needs_by_model[model]
            if not model_needs.index.equals(df.index):
                raise ValueError(f"Need values for {model} must align with the cohort index")
            tolerance = self.allocation_eps
            group = df[self.group_col]
            groups = [[i for i, label in group.items() if label == g] for g in (0, 1)]
            if not groups[0] or not groups[1]:
                raise ValueError("There must be at least 1 individual in each group")

            top_k_idx = df[model].nlargest(self.k).index.to_list()
            top_k_sum = df[model].loc[top_k_idx].sum().item()
            utility_req = top_k_sum - self.k * tolerance

            tables = [
                self.make_group_tables(idx, probabilities, model_needs, min(len(idx), self.k-1))
                for idx in groups
            ]

            def best_pair_for_minimum(minimum):
                best_disparity = np.inf
                best_utility_sum = 0.0
                best_pair = None
                min_count_0 = max(minimum, self.k - len(groups[1]))
                max_count_0 = min(len(groups[0]), self.k - minimum)
                for count_0 in range(min_count_0, max_count_0 + 1):
                    count_1 = self.k - count_0
                    for sum_0, state_0 in tables[0][count_0].items():
                        for sum_1, state_1 in tables[1][count_1].items():
                            utility_sum = state_0[0] + state_1[0]
                            if utility_sum < utility_req:
                                continue

                            if need_scale == 1:
                                disparity = abs(sum_1 / count_1 - sum_0 / count_0)
                            else:
                                # Scaled integer sums represent fractional needs exactly.
                                disparity = abs(sum_1 * count_0 - sum_0 * count_1) / (need_scale * count_0 * count_1)
                            if disparity < best_disparity or (disparity == best_disparity and utility_sum > best_utility_sum):
                                best_disparity = disparity
                                best_utility_sum = utility_sum
                                best_pair = (state_0, state_1)
                return best_pair

            best_pair = best_pair_for_minimum(min_selected_per_group)
            constraint_feasible = best_pair is not None
            if not constraint_feasible and min_selected_per_group > 1:
                # Reuse the group tables; do not rerun OPT or the true-needs DP.
                best_pair = best_pair_for_minimum(1)

            if best_pair is None:
                selected_idx = top_k_idx
            else:
                selected_idx = list(best_pair[0][1] + best_pair[1][1])

            if diagnostics is not None:
                diagnostics.append({
                    "seed": seed, "iteration": iteration, "model": model,
                    "min_selected_per_group": int(min_selected_per_group),
                    "constraint_feasible": constraint_feasible,
                    "fallback_used": not constraint_feasible,
                    "fallback_reason": ("none" if constraint_feasible else
                                        "insufficient_group_members" if min(map(len, groups)) < min_selected_per_group
                                        else "utility_requirement"),
                    "baseline_feasible": best_pair is not None,
                    "selection_source": ("top_k" if best_pair is None else
                                         "constrained_dp" if constraint_feasible and min_selected_per_group > 1
                                         else "baseline_dp"),
                    "available_count_0": len(groups[0]), "available_count_1": len(groups[1]),
                    "selected_count_0": int(group.loc[selected_idx].eq(0).sum()),
                    "selected_count_1": int(group.loc[selected_idx].eq(1).sum()),
                    "utility_sum": float(probabilities.loc[selected_idx].sum()),
                    "utility_requirement": float(utility_req),
                })
            
            k_prime = int(df.loc[selected_idx, self.qualification_col].sum())
            n_prime = int(df[self.qualification_col].sum())
            selected_people = df.loc[selected_idx, "person_id"].to_list()

            selected_people.sort()
            allocation = tuple(selected_people)
            if allocation not in allocations:
                allocation_data.append({
                    "seed": seed,
                    "iteration": iteration,
                    "allocation_idx": allocation_idx,
                    "selected": selected_people,
                    "k'": k_prime,
                    "n'": n_prime,
                })
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": [model]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append(model)

        models = {}
        for v in allocations.values():
            models[v["allocation_idx"]] = v["models"]

        for i in range(len(allocation_data)):
            allocation_data[i]["models"] = models[allocation_data[i]["allocation_idx"]] 

        return allocation_data

    @staticmethod
    def _validate_need_experiment(alpha, beta, replacement_seed):
        for name, value in [("alpha", alpha), ("beta", beta)]:
            if not isinstance(value, Real) or not np.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if not isinstance(replacement_seed, (int, np.integer)) or replacement_seed < 0:
            raise ValueError("replacement_seed must be a nonnegative integer")

    @staticmethod
    def _need_grid_scale(alpha, shrinkage_order, need_step):
        if shrinkage_order not in ("after_rounding", "before_rounding"):
            raise ValueError("shrinkage_order must be 'after_rounding' or 'before_rounding'")
        if (isinstance(need_step, (bool, np.bool_)) or not isinstance(need_step, Real)
                or not np.isfinite(float(need_step)) or not 0 < need_step <= 1):
            raise ValueError("need_step must be a finite positive reciprocal integer, e.g. 1, 0.5, or 0.2")
        inverse_step = 1.0 / float(need_step)
        if not np.isfinite(inverse_step):
            raise ValueError("need_step is too small")
        scale = round(inverse_step)
        if not np.isclose(inverse_step, scale, rtol=0, atol=1e-10):
            raise ValueError("need_step must equal 1 / an integer so integer true needs remain exact")
        if shrinkage_order == "after_rounding" and need_step != 1:
            raise ValueError("need_step other than 1 requires shrinkage_order='before_rounding'")
        if shrinkage_order == "before_rounding" and alpha != 0:
            raise ValueError("The before-rounding experiment requires alpha=0; use after_rounding for replacement")
        return scale

    def _prepare_needs_before_rounding(self, df, perf, exp_need_col, beta, need_scale):
        """Construct scaled DP inputs and measure errors from the same transformation."""
        true_needs = df[self.need_col].to_numpy(dtype=float)
        if not np.isfinite(true_needs).all() or (true_needs != np.floor(true_needs)).any():
            raise ValueError("Before-rounding shrinkage requires finite integer true needs")
        needs_by_model, diagnostics = {}, []
        for model in self.rashomon_models(perf):
            col = self.exp_need_col_for_model(model, exp_need_col)
            predictions = df[col].to_numpy(dtype=float)
            if not np.isfinite(predictions).all():
                raise ValueError(f"Expected needs for {model} must be finite")
            # Preserve endpoints without floating-point cancellation in h + beta * (mu - h).
            if beta == 0:
                adjusted = true_needs.copy()
            elif beta == 1:
                adjusted = predictions.copy()
            else:
                adjusted = true_needs + float(beta) * (predictions - true_needs)
            with np.errstate(over="ignore", invalid="ignore"):
                scaled_needs = np.rint(adjusted * need_scale)
            if not np.isfinite(scaled_needs).all() or (np.abs(scaled_needs) > 2**53).any():
                raise ValueError("Scaled needs exceed exact floating-point integer range; choose a coarser need_step")
            # Python integers also keep DP sums and cross-products free of int64 overflow.
            needs_by_model[model] = pd.Series(
                [int(value) for value in scaled_needs], index=df.index, dtype=object)
            rounded = scaled_needs / need_scale
            row = {"model": model, "n": len(df)}
            for prefix, error in (
                ("prediction", predictions - true_needs),
                ("before_rounding", adjusted - true_needs),
                ("rounding", rounded - adjusted),
                ("after_rounding", rounded - true_needs),
            ):
                row[f"{prefix}_mae"] = float(np.mean(np.abs(error)))
                row[f"{prefix}_rmse"] = float(np.sqrt(np.mean(error**2)))
            diagnostics.append(row)
        return needs_by_model, diagnostics

    @staticmethod
    def _true_need_indices(df, seed, iteration, alpha, replacement_seed):
        """Choose a uniform subset, shared across models and nested across alpha."""
        n_replace = int(alpha * len(df))
        if n_replace == 0:
            return df.index[:0]
        # Sort by identity so an incidental cohort row shuffle cannot change the subset.
        if not df.index.is_unique or not df["person_id"].is_unique:
            raise ValueError("Need replacement requires unique cohort indices and person IDs")
        ordered_index = df.sort_values("person_id").index
        rng = np.random.default_rng(np.random.SeedSequence([replacement_seed, int(seed), int(iteration)]))
        return ordered_index.take(rng.permutation(len(df))[:n_replace])

    def calculate_least_discriminatory_allocation_exp_needs(
        self, seed, iteration, df, perf, exp_need_col="exp_need", *, alpha=0.0, beta=1.0, replacement_seed=0,
        shrinkage_order="after_rounding", need_step=1.0, min_selected_per_group=1,
    ):
        """Run LDA-exp with optional true-need replacement and error shrinkage.

        First form h + beta * (round(predicted_need) - h), then replace the same
        floor(alpha * cohort size) people with h for every model. beta=1 keeps
        the original alpha experiment; alpha=0 runs pure beta shrinkage.
        beta=0 or alpha=1 uses true needs. Neither option changes probabilities.

        For 0 < beta < 1, true needs must be integers. Fractional mixed needs are
        represented by exact integer numerators using beta's decimal fraction;
        they are not rounded back to whole needs. Finer beta fractions may
        increase the number of DP states. Inputs are never modified.

        With shrinkage_order='before_rounding', alpha must be zero. Instead use
        z = h + beta * (predicted_need - h), then round z once to need_step.
        need_step must be 1 / an integer (e.g. 1, 0.5, 0.2). np.rint rounds exact
        halfway values to the nearest even grid index. The DP receives integer
        grid indices with need_scale=1/need_step; there is no second rounding.
        Per-model MAE/RMSE before rounding, from rounding, and after rounding are
        appended to self.need_diagnostics. Probabilities and the Rashomon set
        remain unchanged. The default preserves the original alpha/beta behavior.

        min_selected_per_group requires at least that many selected people in
        each group. If infeasible, fall back to the original minimum-one DP on
        the same transformed needs, including its existing top-k fallback.
        Per-model feasibility, counts, and fallback details are appended to
        self.group_count_diagnostics; the return schema is unchanged.
        """
        self._validate_need_experiment(alpha, beta, replacement_seed)
        self._validate_min_selected_per_group(min_selected_per_group)
        grid_scale = self._need_grid_scale(alpha, shrinkage_order, need_step)
        if not 2 <= self.k <= len(df):
            raise ValueError("Expected-need allocation requires 2 <= k <= cohort size")

        if shrinkage_order == "before_rounding":
            needs_by_model, diagnostics = self._prepare_needs_before_rounding(
                df, perf, exp_need_col, beta, grid_scale)
            if beta == 0:
                # Use the identical arithmetic and tie-breaking as the true-needs baseline.
                result = self.calculate_least_discriminatory_allocation(
                    seed, iteration, df, perf, min_selected_per_group=min_selected_per_group,
                    diagnostics=self.group_count_diagnostics)
            else:
                result = self.calculate_least_discriminatory_allocation(
                    seed, iteration, df, perf, needs_by_model=needs_by_model, need_scale=grid_scale,
                    min_selected_per_group=min_selected_per_group, diagnostics=self.group_count_diagnostics)
            self.need_diagnostics.extend({
                "seed": seed, "iteration": iteration, "alpha": float(alpha), "beta": float(beta),
                "shrinkage_order": shrinkage_order, "need_step": 1.0 / grid_scale,
                "need_scale": grid_scale, **row,
            } for row in diagnostics)
            return result

        true_need_indices = self._true_need_indices(df, seed, iteration, alpha, replacement_seed)
        if beta == 0 or len(true_need_indices) == len(df):
            return self.calculate_least_discriminatory_allocation(
                seed, iteration, df, perf, min_selected_per_group=min_selected_per_group,
                diagnostics=self.group_count_diagnostics)

        beta_fraction = Fraction(1) if beta == 1 else Fraction(str(beta))
        numerator, need_scale = beta_fraction.numerator, beta_fraction.denominator
        if beta != 1:
            true_needs = df[self.need_col].astype(float)
            if not np.isfinite(true_needs).all() or (true_needs != np.floor(true_needs)).any():
                raise ValueError("Beta shrinkage requires finite integer true needs")

        mixed_by_model = {}
        for model in self.rashomon_models(perf):
            col = self.exp_need_col_for_model(model, exp_need_col)
            exp_needs = df[col].astype(float)
            rounded_needs = np.rint(exp_needs).astype("int64")
            if beta != 1:
                # Python integers avoid overflow for beta values with long decimals.
                mixed_needs = pd.Series(
                    [(need_scale - numerator) * int(h) + numerator * int(pred)
                     for h, pred in zip(true_needs, rounded_needs)],
                    index=df.index, dtype=object,
                )
                if len(true_need_indices):
                    mixed_needs.loc[true_need_indices] = true_needs.loc[true_need_indices].map(
                        lambda h: need_scale * int(h))
            elif len(true_need_indices):
                # Round predictions only; preserve the supplied true needs exactly.
                mixed_needs = rounded_needs.astype(float)
                mixed_needs.loc[true_need_indices] = df.loc[true_need_indices, self.need_col]
            else:
                mixed_needs = rounded_needs
            mixed_by_model[model] = mixed_needs

        return self.calculate_least_discriminatory_allocation(
            seed, iteration, df, perf, needs_by_model=mixed_by_model, need_scale=need_scale,
            min_selected_per_group=min_selected_per_group, diagnostics=self.group_count_diagnostics)

    def get_allocations(
        self, exp_need_col=None, *, alpha=0.0, beta=1.0, replacement_seed=0, algorithms=None,
        shrinkage_order="after_rounding", need_step=1.0, min_selected_per_group=1,
    ):
        self._validate_need_experiment(alpha, beta, replacement_seed)
        grid_scale = self._need_grid_scale(alpha, shrinkage_order, need_step)
        if algorithms is None:
            algorithms = ["OPT", "LDA-tru"] + (["LDA-exp"] if exp_need_col is not None else [])
        elif isinstance(algorithms, str):
            algorithms = [algorithms]
        algorithms = set(algorithms)
        if not algorithms or not algorithms <= {"OPT", "LDA-tru", "LDA-exp"}:
            raise ValueError("algorithms must contain OPT, LDA-tru, and/or LDA-exp")
        if "LDA-exp" in algorithms and exp_need_col is None:
            raise ValueError("exp_need_col is required for LDA-exp")
        if (alpha != 0 or beta != 1) and "LDA-exp" not in algorithms:
            raise ValueError("alpha and beta experiments require LDA-exp")
        if shrinkage_order == "before_rounding" and "LDA-exp" not in algorithms:
            raise ValueError("Before-rounding experiments require LDA-exp")
        if "LDA-exp" in algorithms or min_selected_per_group != 1:
            self._validate_min_selected_per_group(min_selected_per_group)
        if min_selected_per_group != 1 and "LDA-exp" not in algorithms:
            raise ValueError("Minimum selected group count experiments require LDA-exp")
        self.need_diagnostics = []
        self.group_count_diagnostics = []
        allocations_opt = []
        allocations_lda_tru = []
        allocations_lda_exp = []
        people = []
        rng = np.random.default_rng(0)
        for seed in range(self.n_seeds):
            df_seed = self.df[(self.df["seed"]==seed)&(self.df["person_id"]>=0)].copy()
            df_perf = self.df[(self.df["seed"]==seed)&(self.df["person_id"]<0)].copy()
        
            for i in tqdm(range(self.iters_per_seed), desc=f"[Seed {seed}]"):
                k_0 = rng.choice(range(1, self.sample_size))
                k_1 = self.sample_size - k_0
                sample_g0 = df_seed[df_seed[self.group_col]==0].sample(n=k_0, random_state=i)
                sample_g1 = df_seed[df_seed[self.group_col]==1].sample(n=k_1, random_state=i)
                people_sample = pd.concat((sample_g0, sample_g1)).sample(frac=1, random_state=i, ignore_index=True)
                # people_sample = split_df.sample(n=self.TEST_SIZE, random_state=i)
                true_need_indices = self._true_need_indices(people_sample, seed, i, alpha, replacement_seed)
                people.append({
                    "seed": seed,
                    "iteration": i,
                    "people": people_sample["person_id"].to_list(),
                    "alpha": alpha,
                    "beta": float(beta),
                    "replacement_seed": replacement_seed,
                    "true_need_person_ids": people_sample.loc[true_need_indices, "person_id"].to_list(),
                })
                if shrinkage_order == "before_rounding":
                    people[-1].update({
                        "shrinkage_order": shrinkage_order,
                        "need_step": 1.0 / grid_scale,
                        "need_scale": grid_scale,
                    })
                if min_selected_per_group != 1:
                    people[-1]["min_selected_per_group"] = int(min_selected_per_group)
                
                if "OPT" in algorithms:
                    allocations_opt += self.calculate_topk_allocations(seed, i, people_sample, df_perf)
                if "LDA-tru" in algorithms:
                    allocations_lda_tru += self.calculate_least_discriminatory_allocation(seed, i, people_sample, df_perf)
                if "LDA-exp" in algorithms:
                    allocations_lda_exp += self.calculate_least_discriminatory_allocation_exp_needs(
                        seed, i, people_sample, df_perf,
                        exp_need_col=exp_need_col,
                        alpha=alpha,
                        beta=beta,
                        replacement_seed=replacement_seed,
                        shrinkage_order=shrinkage_order,
                        need_step=need_step,
                        min_selected_per_group=min_selected_per_group,
                    )
        
        df_people = pd.DataFrame(people)
        df_allocations_opt = pd.DataFrame(allocations_opt)
        df_allocations_lda_tru = pd.DataFrame(allocations_lda_tru)
        df_allocations_lda_exp = pd.DataFrame(allocations_lda_exp)
        return df_allocations_opt, df_allocations_lda_tru, df_allocations_lda_exp, df_people
