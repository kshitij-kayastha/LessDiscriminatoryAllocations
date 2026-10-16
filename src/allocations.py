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

    def calculate_rashomon_allocations(self, seed, iteration, df, perf):
        best_perf = 1
        for model in range(1, self.NUM_MODELS+1):
            model_perf = float(perf.loc[perf["person_id"]==-1, "m_"+str(model)].item())
            best_perf = min(best_perf, model_perf)

        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model in range(1, self.NUM_MODELS+1):
            if float(perf.loc[perf["person_id"]==-1, "m_"+str(model)].item()) > best_perf + self.RASHOMON_EPSILON:
                continue

            selected_idx = df["m_"+str(model)].nlargest(self.k).index.to_list()
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
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": ["m_"+str(model)]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append("m_"+str(model))

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

    def calculate_least_discriminatory_allocation(self, seed, iteration, df, perf, needs=None):
        """Run the DP using actual needs, or an explicitly supplied need Series."""
        if needs is None:
            needs = df[self.NEED_COLUMN]

        best_perf = 1
        for model in range(1, self.NUM_MODELS+1):
            model_perf = float(perf.loc[perf["person_id"]==-1, "m_"+str(model)].item())
            best_perf = min(best_perf, model_perf)

        allocations = {}
        allocation_data = []
        allocation_idx = 0
        for model in range(1, self.NUM_MODELS+1):
            if float(perf.loc[perf["person_id"]==-1, "m_"+str(model)].item()) > best_perf + self.RASHOMON_EPSILON:
                continue

            probabilities = df["m_"+str(model)]
            tolerance = self.ALLOCATION_EPSILON
            group = df[self.GROUP_COLUMN]
            groups = [[i for i, label in group.items() if label == g] for g in (0, 1)]
            if not groups[0] or not groups[1]:
                raise ValueError("There must be at least 1 individual in each group")

            top_k_idx = df["m_"+str(model)].nlargest(self.k).index.to_list()
            top_k_sum = df["m_"+str(model)].loc[top_k_idx].sum().item()
            utility_req = top_k_sum - self.k * tolerance

            tables = [
                self.table_group(idx, probabilities, needs, min(len(idx), self.k-1))
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
                allocations[allocation] = {"allocation_idx": allocation_idx, "models": ["m_"+str(model)]}
                allocation_idx += 1
            else:
                allocations[allocation]["models"].append("m_"+str(model))

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

        expected_needs = df[expected_need_column].astype(float)
        if not np.isfinite(expected_needs).all() or (expected_needs < 0).any():
            raise ValueError("Expected needs must be finite and nonnegative")

        max_need = expected_needs.max()
        if need_bound is None:
            need_bound = max_need
        if not np.isfinite(need_bound) or need_bound < max_need:
            raise ValueError("need_bound must be finite and at least the largest expected need")

        if max_need == 0:
            rounded_needs = pd.Series(0, index=df.index, dtype="int64")
        else:
            b = eta * need_bound / 2
            if not np.isfinite(b) or b <= 0:
                raise ValueError("eta and need_bound must give a finite, positive rounding step")
            scaled_needs = expected_needs / b
            if not np.isfinite(scaled_needs).all():
                raise ValueError("eta is too small for integer DP need sums; increase eta")
            rounded_needs = np.floor(scaled_needs).astype("int64")

        return self.calculate_least_discriminatory_allocation(
            seed, iteration, df, perf, needs=rounded_needs,
        )

    def get_allocations(self, expected_need_column=None, eta=0.1, need_bound=None):
        allocations_opt = []
        allocations_lda = []
        people = []
        for split in range(self.NUM_SPLITS):
            split_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]>=0)].copy()
            perf_df = self.df[(self.df["seed"]==split)&(self.df["person_id"]<0)].copy()
        
            for i in tqdm(range(self.ITERATIONS_PER_SPLIT), desc=f"[Split {split}]"):
                k_0 = np.random.choice(range(1, self.TEST_SIZE))
                k_1 = self.TEST_SIZE - k_0
                sample_0 = split_df[split_df[self.GROUP_COLUMN]==0].sample(n=k_0, random_state=i)
                sample_1 = split_df[split_df[self.GROUP_COLUMN]==1].sample(n=k_1, random_state=i)
                people_sample = pd.concat((sample_0, sample_1)).sample(frac=1, ignore_index=True)
                # people_sample = split_df.sample(n=self.TEST_SIZE, random_state=i)
                people.append({"seed": split, "iteration": i, "people": people_sample["person_id"].to_list()})
                
                allocations_opt += self.calculate_rashomon_allocations(split, i, people_sample, perf_df)
                if expected_need_column is None:
                    allocations_lda += self.calculate_least_discriminatory_allocation(split, i, people_sample, perf_df)
                else:
                    allocations_lda += self.calculate_least_discriminatory_allocation_expected_needs(
                        split, i, people_sample, perf_df,
                        expected_need_column=expected_need_column,
                        eta=eta,
                        need_bound=need_bound,
                    )
        people = pd.DataFrame(people)
        allocations_opt = pd.DataFrame(allocations_opt)
        allocations_lda = pd.DataFrame(allocations_lda)
        return allocations_opt, allocations_lda, people
