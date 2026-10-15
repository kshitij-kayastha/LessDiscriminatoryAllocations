### Methods used Allocation Multiplicity Paper

Four notebooks correspond to the paper's four model-generation methods (some saved settings differ from the paper)

| Method in the paper | Prediction notebook | What the code does |
|---|---|---|
| **Different Feature Subsets** | [predictions_faster_risk.ipynb](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/predictions_faster_risk.ipynb) | Uses FasterRisk to generate a pool of sparse scoring models with integer coefficients. Models vary in their selected features and coefficients. |
| **Bootstrapped Training Data** | [predictions_bootstrap.ipynb](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/predictions_bootstrap.ipynb) | Trains models on different randomly selected training subsets. Supports logistic regression, random forests, and neural networks. Use `MODEL_CLASS = "nn"` to match the model family described in the paper. |
| **Shuffled Data Order** | [predictions_nn_variance.ipynb](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/predictions_nn_variance.ipynb) | Trains neural networks and saves predictions at successive training epochs, treating each checkpoint as a candidate model. **The inspected code does not explicitly shuffle training data.** |
| **Weight Perturbation** | [predictions_awp.ipynb](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/predictions_awp.ipynb) | Trains a base neural network, then repeatedly copies and modifies it to increase the qualification prediction for a particular validation individual, subject to a validation-loss stopping condition. |

Some implementation details that need to be changed for our experiments.
1. **The bootstrap notebook currently runs logistic regression.**<br>
Its saved configuration is `MODEL_CLASS = "lr"`, with 2 data splits and 100 candidate models per split. Furthermore, its sampling uses a random permutation followed by selecting half the training observations. That is sampling without replacement, despite the bootstrap name. Preserving their implementation means retaining this behavior. 

2. **The "Shuffled Data Order" mapping is the indended correspondence, rather than an exact description of the current implementation.**<br>
The visualization notebooks associate `nn_variance` with model initialization. The prediction notebook uses full-batch updates and saves checkpoints at epochs 200–299 across 10 training runs, producing 1,000 candidate prediction columns per split. I found no data-shuffling step. It also seeds each run with `iteration * random_seed`; when the split seed is zero, every run receives the same initialization seed.

3. **The FasterRisk notebook is configured for a smaller search.**<br>
Currently, it uses `sparsity = [5]`, `coeff_bounds = [5]`, and 2 splits. Larger parameter grids appear in comments. Running the notebook unchanged therefore should not be assumed to reproduce the paper’s reported model counts.

4. **The paper’s neural weight-perturbation method is AWP, not LWP.**<br>
The additional [predictions_lwp.ipynb](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/predictions_lwp.ipynb) fits a linear regression model and randomly perturbs its coefficients. It does not match the neural-network perturbation method described in the paper. Its raw linear predictions are also not guaranteed to lie in \([0,1]\), so they cannot automatically be treated as qualification probabilities.

---

The notebooks generate candidate models, the filtering for models belonging to the Rashomon set happens afterward. 

Their [allocation.py](https://github.com/mqwangwa/allocation_multiplicity/blob/c7428c094ade9dc1cf5a6fd7870390cdc9531199/allocations.py) reads the validation loss and retains models satisfying

$$
    L_{val}(m) \le \min_{m'} L_{val}(m') + \mathrm{RASHOMON\_EPSILON}
$$

Each prediction file stores:
- `m_1`, `m_2`, ... candidate model predictions.
- `person_id == -1`: validation losses.
- `person_id == -2`: training losses.
- `seed`: the data split
