import numpy as np
import pandas as pd
from scipy.stats import differential_entropy, entropy

# Metrics calculated over Rashomon Allocations 
# for a fixed test size, selection rate, and qualification rate
# computed over each seed / iteration 

class Metrics:

	# df = allocations dataframe
	# people = test set dataframe
	# data = original dataset of features
	def __init__(self, df, people_df, data, QUALIFICATION_COLUMN):
		self.df = df
		self.people_df = people_df
		self.data = data 
		self.QUALIFICATION_COLUMN = QUALIFICATION_COLUMN
		self.k = len(self.df.loc[0, "selected"])
		self.n = len(self.people_df.loc[0, "people"])


	# Number of qualified people selected (k')
	def k_prime(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				metric.append(self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean())

		return np.mean(metric), np.std(metric)

	# Number of qualified people in test set (n')
	def n_prime(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				metric.append(self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean())

		return np.mean(metric), np.std(metric)

	# Number of Rashomon allocations found
	def count_rashomon_allocations(self):
		grouped = self.df.groupby(["seed", "iteration"]).count()["allocation_idx"].reset_index()
		return np.mean(grouped["allocation_idx"]), np.std(grouped["allocation_idx"])

	# Number of Rashomon models found
	def count_rashomon_models(self):
		grouped = self.df.groupby(["seed", "iteration"]).sum(numeric_only=True)["model_count"].reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])

	# Number of Rashomon models per Rashomon allocation
	def count_rashomon_models_per_allocation(self):
		grouped = self.df.groupby(["seed", "iteration"])["model_count"].mean().reset_index()
		return np.mean(grouped["model_count"]), np.std(grouped["model_count"])


	# Homogenization -- Systemic Rejection: Number of people never selected across Rashomon allocations
	def systemic_exclusion(self):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				# Calculate union of selected people
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				selected = set(allocations[0])
				for a in allocations:
					selected = selected.union(set(a))
				metric.append((self.n-len(selected))/self.n)
		return np.mean(metric), np.std(metric)



	def shannon_entropy(self, probs):
		e = 0
		for p in probs:
			if p > 0 and p < 1:
				e += p * np.log2(p)
		return -e

	# Helper functions for entropy-based homogenization measures
	def heterozygosity(self, probs):
		e = 0
		for p in probs:
			e += p ** 2
		return 1-e

	def calculate_entropy(self, values, type="shannon"):
		if type=="shannon":
			_, counts = np.unique(values, return_counts=True)
			probabilities = counts / counts.sum()
			return entropy(probabilities, base=2)
		elif type=="differential":
			return differential_entropy(values)
		else:
			return np.nan

	# Homogenization -- Heterozygosity in individual decisions over Rashomon allocations
	# Baseline: Entropy using expected selection rate, based on qualification and precision
	def homogenization_in_individual_decisions(self):
		metric = []
		baseline = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				people = list(self.people_df.loc[((self.people_df["seed"]==seed)&(self.people_df["iteration"]==iteration)), "people"])[0]
				selected_counts = dict.fromkeys(people, 0)

				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				for a in allocations:
					for p in a:
						selected_counts[p] += 1

				metric.append(np.mean([self.heterozygosity([value / len(allocations), 1 - (value / len(allocations))]) for value in selected_counts.values()]))

				# Baseline
				n_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean()
				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()
				qualified_prob = k_prime / n_prime
				qualified_entropy = self.heterozygosity([qualified_prob, 1 - qualified_prob])
				unqualified_prob = (self.k - k_prime) / (self.n - n_prime)
				unqualified_entropy = self.heterozygosity([unqualified_prob, 1-unqualified_prob])
				baseline.append((qualified_entropy * (n_prime / self.n)) + (unqualified_entropy * ((self.n-n_prime)/self.n)))
				
		return np.mean(metric), np.std(metric), np.mean(baseline), np.std(baseline)

	# Homogenization -- Entropy in feature over selected individuals
	# Baseline 1: Entropy in feature over all qualified individuals
	# Baseline 2: Weighted average of entropy over qualified and unqualified individuals, based on precision
	def homogenization_in_selected_individuals(self, feature, entropy_type="shannon"):
		metric = []
		baseline1 = []
		baseline2 = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				people = list(self.people_df.loc[((self.people_df["seed"]==seed)&(self.people_df["iteration"]==iteration)), "people"])[0]
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", self.QUALIFICATION_COLUMN, feature]]
				
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					metric_inner.append(self.calculate_entropy(test_data.loc[test_data["person_id"].isin(a), feature].to_list(), entropy_type))
				metric.append(np.mean(metric_inner))


				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()

				qualified_entropy = self.calculate_entropy(test_data.loc[test_data[self.QUALIFICATION_COLUMN]==1, feature].to_list(), entropy_type)
				baseline1.append(qualified_entropy)

				unqualified_entropy = self.calculate_entropy(test_data.loc[test_data[self.QUALIFICATION_COLUMN]==0, feature].to_list(), entropy_type)
				baseline2.append((qualified_entropy * (k_prime / self.k)) + (unqualified_entropy * ((self.k-k_prime)/self.k)))

		return np.mean(metric), np.std(metric), np.mean(baseline1), np.std(baseline1), np.mean(baseline2), np.std(baseline2)


	# Group Fairness -- selection rate for demographic group over Rashomon Allocations
	# Among the selection rates found, returns the average and best
	def group_selection_rates_found(self, group_col, group_val):
		metric = []
		group_people = self.data.loc[self.data[group_col]==group_val, "person_id"].to_list()

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					group_selected = 0
					for p in a:
						if p in group_people:
							group_selected += 1
					metric_inner.append(group_selected/len(a))
				metric.append(metric_inner)

		metric_avg = [np.nanmean(inner) for inner in metric]
		metric_best = [np.nanmax(inner) for inner in metric]
		return np.nanmean(metric_avg), np.nanstd(metric_avg), np.nanmean(metric_best), np.nanstd(metric_best)

	# Group Fairness -- selection rate for demographic group over Rashomon Allocations
	# Among the selection rates possible, returns the average and best
	def group_selection_rates_possible(self, group_col, group_val):
		metric_avg = []
		metric_best = []

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				people = list(self.people_df.loc[((self.people_df["seed"]==seed)&(self.people_df["iteration"]==iteration)), "people"])[0]
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", self.QUALIFICATION_COLUMN, group_col]]

				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()
				n_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean()

				group_qualified = len(test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==1)])
				group_unqualified = len(test_data[(test_data[group_col]==group_val)&(test_data[self.QUALIFICATION_COLUMN]==0)])

				metric_avg.append(((group_qualified / n_prime) * (k_prime / self.k)) + ((group_unqualified / (self.n-n_prime)) * ((self.k-k_prime)/self.k)))

				best_inner = []
				for k_prime in range(self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].min(), self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].max() + 1):
					best_inner.append((min(k_prime, group_qualified) + min(self.k-k_prime, group_unqualified))/self.k)
				metric_best.append(max(best_inner))

		return np.mean(metric_avg), np.std(metric_avg), np.mean(metric_best), np.std(metric_best)


	# Group Fairness -- Among selected, check ratio of a feature by group
	# Assumes denominator group will have smaller value of the feature
	def group_feature_ratios_found(self, group_col, group_val_num, group_val_denom, feature_col):
		metric = []
		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				people = list(self.people_df.loc[((self.people_df["seed"]==seed)&(self.people_df["iteration"]==iteration)), "people"])[0]
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", group_col, feature_col]]

				allocations = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "selected"].to_list()
				metric_inner = []
				for a in allocations:
					feature_group_num = test_data.loc[(test_data[group_col]==group_val_num)&(test_data["person_id"].isin(a)), feature_col].mean()
					feature_group_denom = test_data.loc[(test_data[group_col]==group_val_denom)&(test_data["person_id"].isin(a)), feature_col].mean()
					metric_inner.append(feature_group_num / feature_group_denom)
				metric.append(metric_inner)

		metric_avg = [np.nanmean(inner) if not np.all(np.isnan(inner)) else np.nan for inner in metric]
		metric_best = [np.nanmin(inner) if not np.all(np.isnan(inner)) else np.nan for inner in metric]
		return np.nanmean(metric_avg), np.nanstd(metric_avg), np.nanmean(metric_best), np.nanstd(metric_best)

	# Group Fairness -- Among selected, what is the best possible ratio of a feature by group
	def group_feature_ratios_possible(self, group_col, group_val_num, group_val_denom, feature_col):
		# Helper function to add rank variable for alternating sort
		def add_rank(group):
			group["rank"] = range(len(group))
			return group

		metric_avg = []
		metric_best = []

		for seed in self.df["seed"].unique():
			for iteration in self.df["iteration"].unique():
				people = list(self.people_df.loc[((self.people_df["seed"]==seed)&(self.people_df["iteration"]==iteration)), "people"])[0]
				test_data = self.data.loc[self.data["person_id"].isin(people), ["person_id", self.QUALIFICATION_COLUMN, group_col, feature_col]]

				k_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "k'"].mean()
				n_prime = self.df.loc[(self.df["seed"]==seed)&(self.df["iteration"]==iteration), "n'"].mean()

				# Number of qualified and unqualified people in each group
				g1_q = len(test_data[(test_data[group_col]==group_val_num)&(test_data[self.QUALIFICATION_COLUMN]==1)])
				g1_uq = len(test_data[(test_data[group_col]==group_val_num)&(test_data[self.QUALIFICATION_COLUMN]==0)])
				g2_q = len(test_data[(test_data[group_col]==group_val_denom)&(test_data[self.QUALIFICATION_COLUMN]==1)])
				g2_uq = len(test_data[(test_data[group_col]==group_val_denom)&(test_data[self.QUALIFICATION_COLUMN]==0)])

				# Average feature value of qualified and unqualified people in each group
				g1_q_f = test_data.loc[(test_data[group_col]==group_val_num)&(test_data[self.QUALIFICATION_COLUMN]==1), feature_col].mean() if g1_q > 0 else 0
				g1_uq_f = test_data.loc[(test_data[group_col]==group_val_num)&(test_data[self.QUALIFICATION_COLUMN]==0), feature_col].mean() if g1_uq > 0 else 0
				g2_q_f = test_data.loc[(test_data[group_col]==group_val_denom)&(test_data[self.QUALIFICATION_COLUMN]==1), feature_col].mean() if g2_q > 0 else 0
				g2_uq_f = test_data.loc[(test_data[group_col]==group_val_denom)&(test_data[self.QUALIFICATION_COLUMN]==0), feature_col].mean() if g2_uq > 0 else 0

				# Average number of selections for qualified and unqualified people in each group
				g1_q_w = k_prime * (g1_q / (g1_q + g2_q))
				g1_uq_w = (self.k-k_prime) * (g1_uq / (g1_uq + g2_uq))
				g2_q_w = k_prime * (g2_q / (g1_q + g2_q))
				g2_uq_w = (self.k-k_prime) * (g2_uq / (g1_uq + g2_uq))

				# Expected feature value for each group, based on qualified and unqualified selections
				g1_f = g1_q_f * (g1_q_w / (g1_q_w + g1_uq_w)) + g1_uq_f * (g1_uq_w / (g1_q_w + g1_uq_w))
				g2_f = g2_q_f * (g2_q_w / (g2_q_w + g2_uq_w)) + g2_uq_f * (g2_uq_w / (g2_q_w + g2_uq_w))
				metric_avg.append(g1_f / g2_f) 

				####
				# qualified_people = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)].groupby([feature_col, group_col], group_keys=True, as_index=False).apply(add_rank)
				# qualified_people = qualified_people.sort_values(by=[feature_col, "rank"], ascending=[False, True]).drop(columns="rank").reset_index(drop=True)
				# unqualified_people = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)].groupby([feature_col, group_col], group_keys=False).apply(add_rank)
				# unqualified_people = unqualified_people.sort_values(by=[feature_col, "rank"], ascending=[False, True]).drop(columns="rank").reset_index(drop=True)
				
				qualified_people = test_data[(test_data[self.QUALIFICATION_COLUMN]==1)].copy()
				qualified_people["rank"] = qualified_people.groupby([feature_col, group_col]).cumcount()
				qualified_people = qualified_people.sort_values(by=[feature_col, "rank"], ascending=[False, True]).drop(columns="rank").reset_index(drop=True)
				unqualified_people = test_data[(test_data[self.QUALIFICATION_COLUMN]==0)].copy()
				unqualified_people["rank"] = unqualified_people.groupby([feature_col, group_col]).cumcount()
				unqualified_people = unqualified_people.sort_values(by=[feature_col, "rank"], ascending=[False, True]).drop(columns="rank").reset_index(drop=True)

				k_prime = int(np.round(k_prime))
				selected = pd.concat([qualified_people.loc[:k_prime], unqualified_people.loc[:self.k-k_prime]])
				g1_f = np.mean(selected.loc[selected[group_col]==group_val_num, feature_col])
				g2_f = np.mean(selected.loc[selected[group_col]==group_val_denom, feature_col])
				metric_best.append(g1_f / g2_f)

		return np.nanmean(metric_avg), np.nanstd(metric_avg), np.nanmean(metric_best), np.nanstd(metric_best)

