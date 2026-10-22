import numpy as np
import pandas as pd
from tqdm import tqdm


class Allocations:
    def __init__(self, df, TEST_SIZE, SELECTION_RATE, QUALIFICATION_COLUMN, NEED_COLUMN, GROUP_COLUMN, RASHOMON_EPSILON, ALLOCATION_EPSILON, NUM_MODELS=None, ITERATIONS_PER_SPLIT=5):
        self.df = df
        self.TEST_SIZE = TEST_SIZE
        self.SELECTION_RATE = SELECTION_RATE
        self.k = int(self.SELECTION_RATE*self.TEST_SIZE)
        self.QUALIFICATION_COLUMN = QUALIFICATION_COLUMN
        self.NEED_COLUMN = NEED_COLUMN
        self.GROUP_COLUMN = GROUP_COLUMN
        self.RASHOMON_EPSILON = RASHOMON_EPSILON
        self.ALLOCATION_EPSILON = ALLOCATION_EPSILON
        self.NUM_MODELS = NUM_MODELS if NUM_MODELS else len([c for c in self.df.columns if c.startswith("m_")])
        self.NUM_SPLITS = len(self.df["seed"].unique())
        self.ITERATIONS_PER_SPLIT = ITERATIONS_PER_SPLIT

    def _eligible_models(self, perf):
        names = [f"m_{i}" for i in range(1, self.NUM_MODELS + 1)]
        rows = perf.loc[perf["person_id"] == -1, names]
        if len(rows) != 1:
            raise ValueError("Expected exactly one validation-loss row per split")
        losses = rows.iloc[0].astype(float)
        if not np.isfinite(losses).all():
            raise ValueError("Validation losses must be finite")
        return losses.index[losses <= losses.min() + self.RASHOMON_EPSILON].tolist()

    @staticmethod
    def expected_need_column_for_model(model_name, expected_need_column="expected_need_{model}"):
        if not isinstance(expected_need_column, str) or not expected_need_column:
            raise ValueError("expected_need_column must be a column name or template")
        try:
            return expected_need_column.format(model=model_name)
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError("Use {model} as the placeholder, e.g. expected_need_{model}") from error

    def calculate_rashomon_allocations(self, seed, iteration, df, perf):
        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model_name in self._eligible_models(perf):

            selected_idx = df[model_name].nlargest(self.k).index.to_list()
            k_prime = int(df.loc[selected_idx, self.QUALIFICATION_COLUMN].sum())
            n_prime = int(df[self.QUALIFICATION_COLUMN].sum())
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
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": [model_name]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append(model_name)

        models = {}
        for v in allocations.values():
            models[v["allocation_idx"]] = v["models"]

        for i in range(len(allocation_data)):
            allocation_data[i]["models"] = models[allocation_data[i]["allocation_idx"]] 

        return allocation_data

    def table_group(self, selected_idx, probabilities, needs, max_count):
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

    def calculate_least_discriminatory_allocation(
        self, seed, iteration, df, perf, needs=None, *, needs_by_model=None,
    ):
        if needs is not None and needs_by_model is not None:
            raise ValueError("Pass either shared needs or needs_by_model, not both")
        if needs is None and needs_by_model is None:
            needs = df[self.NEED_COLUMN]

        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model_name in self._eligible_models(perf):

            probabilities = df[model_name]
            model_needs = needs if needs_by_model is None else needs_by_model[model_name]
            if not model_needs.index.equals(df.index):
                raise ValueError(f"Need values for {model_name} must align with the cohort index")
            tolerance = self.ALLOCATION_EPSILON
            group = df[self.GROUP_COLUMN]
            groups = [[i for i, label in group.items() if label == g] for g in (0, 1)]
            if not groups[0] or not groups[1]:
                raise ValueError("There must be at least 1 individual in each group")

            top_k_idx = df[model_name].nlargest(self.k).index.to_list()
            top_k_sum = df[model_name].loc[top_k_idx].sum().item()
            utility_req = top_k_sum - self.k * tolerance

            tables = [
                self.table_group(idx, probabilities, model_needs, min(len(idx), self.k-1))
                for idx in groups
            ]

            best_disparity = np.inf
            best_utility_sum = 0.0
            best_pair = None

            min_count_0 = max(1, self.k - len(groups[1]))
            max_count_0 = min(len(groups[0]), self.k - 1)

            for count_0 in range(min_count_0, max_count_0 + 1):
                count_1 = self.k - count_0
                for sum_0, state_0 in tables[0][count_0].items():
                    for sum_1, state_1 in tables[1][count_1].items():
                        utility_sum = state_0[0] + state_1[0]
                        if utility_sum < utility_req:
                            continue

                        disparity = abs(sum_1 / count_1 - sum_0 / count_0)
                        if best_disparity is None or disparity < best_disparity or (disparity == best_disparity and utility_sum > best_utility_sum):
                            best_disparity = disparity
                            best_utility_sum = utility_sum
                            best_pair = (state_0, state_1)

            if best_pair is None:
                selected_idx = top_k_idx
            else:
                selected_idx = list(best_pair[0][1] + best_pair[1][1])
            
            k_prime = int(df.loc[selected_idx, self.QUALIFICATION_COLUMN].sum())
            n_prime = int(df[self.QUALIFICATION_COLUMN].sum())
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
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": [model_name]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append(model_name)

        models = {}
        for v in allocations.values():
            models[v["allocation_idx"]] = v["models"]

        for i in range(len(allocation_data)):
            allocation_data[i]["models"] = models[allocation_data[i]["allocation_idx"]] 

        return allocation_data

    def calculate_least_discriminatory_allocation_expected_needs(self, seed, iteration, df, perf, expected_need_column="expected_need", eta=0.1, need_bound=None):
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not 2 <= self.k <= len(df):
            raise ValueError("Expected-need allocation requires 2 <= k <= cohort size")

        rounded_by_model = {}
        for model_name in self._eligible_models(perf):
            column = self.expected_need_column_for_model(model_name, expected_need_column)
            if column not in df.columns:
                raise ValueError(f"Missing expected-need column {column!r} for {model_name}")
            expected_needs = df[column].astype(float)
            if not np.isfinite(expected_needs).all() or (expected_needs < 0).any():
                raise ValueError(f"Expected needs in {column!r} must be finite and nonnegative")

            max_need = expected_needs.max()
            model_bound = max_need if need_bound is None else need_bound
            if not np.isfinite(model_bound) or model_bound < max_need:
                raise ValueError(f"need_bound must be finite and at least the largest value in {column!r}")

            if max_need == 0:
                rounded_needs = pd.Series(0, index=df.index, dtype="int64")
            else:
                rounded_needs = np.floor(expected_needs).astype("int64")
            rounded_by_model[model_name] = rounded_needs

        return self.calculate_least_discriminatory_allocation(
            seed, iteration, df, perf, needs_by_model=rounded_by_model,
        )

    def get_allocations(self, expected_need_column=None, eta=0.1, need_bound=None):
        allocations_opt = []
        allocations_lda_tru = []
        allocations_lda_exp = []
        people = []
        rng = np.random.default_rng(0)
        for split in range(self.NUM_SPLITS):
            split_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]>=0)].copy()
            perf_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]<0)].copy()
        
            for i in tqdm(range(self.ITERATIONS_PER_SPLIT), desc=f"[Split {split}]"):
                k_0 = rng.choice(range(1, self.TEST_SIZE))
                k_1 = self.TEST_SIZE - k_0
                sample_0 = split_df[split_df[self.GROUP_COLUMN]==0].sample(n=k_0, random_state=i)
                sample_1 = split_df[split_df[self.GROUP_COLUMN]==1].sample(n=k_1, random_state=i)
                people_sample = pd.concat((sample_0, sample_1)).sample(frac=1, ignore_index=True)
                # people_sample = split_df.sample(n=self.TEST_SIZE, random_state=i)
                people.append({"seed": split, "iteration": i, "people": people_sample["person_id"].to_list()})
                
                allocations_opt += self.calculate_rashomon_allocations(split, i, people_sample, perf_df)
                allocations_lda_tru += self.calculate_least_discriminatory_allocation(split, i, people_sample, perf_df)
                if expected_need_column is not None:
                    allocations_lda_exp += self.calculate_least_discriminatory_allocation_expected_needs(
                        split, i, people_sample, perf_df,
                        expected_need_column=expected_need_column,
                        eta=eta,
                        need_bound=need_bound,
                    )
                else:
                    allocations_lda_exp = []
        people = pd.DataFrame(people)
        allocations_opt = pd.DataFrame(allocations_opt)
        allocations_lda_tru = pd.DataFrame(allocations_lda_tru)
        allocations_lda_exp = pd.DataFrame(allocations_lda_exp)
        return allocations_opt, allocations_lda_tru, allocations_lda_exp, people
