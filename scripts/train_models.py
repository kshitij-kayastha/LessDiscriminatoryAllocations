"""Train the Rashomon pool of multiclass need models.

A command-line port of test/01-predictions_bootstrap_exp_needs_multiclass.ipynb,
with three changes:

  * the model loop runs in parallel, mirroring allocations_parallel.py's use of
    loky with one BLAS thread per worker;
  * BOTH probability methods come out of a single training run. In the notebook
    PRED_METHOD only renames the output file -- the loop never forwards it to
    derive_predictions -- so an "argmax" run silently writes averages. Here each
    model's class distribution yields both, so the grid halves;
  * --method dispatches the model-generation procedure, ready for the other
    three methods from the paper. Only bootstrap is wired up so far.

Output paths, column names, sentinel rows and dtypes match the notebook exactly.

    uv run python scripts/train_models.py --need-threshold 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

ROOT = Path(__file__).resolve().parent.parent

# --- from the notebook's config cell -------------------------------------
N_SEEDS = 5
N_MODELS = 1000
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
BOOTSTRAP_SIZE = 0.50
NEED_COL = "gagne_sum_t"
MODEL_CLASS = "nn_multiclass"

FEATURES = [
    'dem_female', 'dem_age_band_18-24_tm1', 'dem_age_band_25-34_tm1',
    'dem_age_band_35-44_tm1', 'dem_age_band_45-54_tm1',
    'dem_age_band_55-64_tm1', 'dem_age_band_65-74_tm1', 'dem_age_band_75+_tm1',
    'hypertension_elixhauser_tm1', 'cost_dialysis_tm1', 'cost_emergency_tm1',
    'cost_home_health_tm1', 'cost_ip_medical_tm1', 'cost_ip_surgical_tm1',
    'cost_laboratory_tm1', 'cost_op_primary_care_tm1', 'cost_op_specialists_tm1',
    'cost_op_surgery_tm1', 'cost_other_tm1', 'cost_pharmacy_tm1',
    'cost_physical_therapy_tm1', 'cost_radiology_tm1', 'gagne_sum_tm1',
]
OTHER_VARIABLES = ['person_id', 'gagne_sum_t', 'cost_t']

PRED_METHODS = ("average", "argmax")
METHODS = ("bootstrap",)   # awp, nn_variance, fasterrisk to follow


def validate_needs(values, bound):
    values = np.asarray(values, dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any() or (values > bound).any()
            or (values != np.floor(values)).any()):
        raise ValueError(f'Need labels must be finite integers in 0 - {bound}.')
    return values.astype(np.int64)


class MulticlassNeedMLP(MLPClassifier):
    def __init__(self, need_bound, random_state=None, max_iter=1000):
        self.need_bound = need_bound
        # No hidden_layer_sizes: sklearn's default is (100,). The larger
        # (50, 100, 200) network did not lower validation loss, and it widened
        # the spread of those losses from about 0.008 to about 0.17, which left
        # a Rashomon set of 4-15 models out of 1000 at eps = 0.01.
        super().__init__(random_state=random_state, max_iter=max_iter)

    def _validate_input(self, X, y, incremental, reset):
        # Retain sklearn's input validation; replace only the target encoding.
        X, _ = super()._validate_input(X, y, incremental, reset)
        labels = validate_needs(y, self.need_bound)
        self._label_binarizer = LabelBinarizer().fit(np.arange(self.need_bound + 1))
        self.classes_ = self._label_binarizer.classes_
        return X, self._label_binarizer.transform(labels).astype(bool)


def build_model(method, X_train, h_train, index, need_bound):
    """One model from the pool. `index` is the model number, not the split seed."""
    if method == "bootstrap":
        np.random.seed(index)
        idx = np.random.permutation(len(X_train))[:int(len(X_train) * BOOTSTRAP_SIZE)]
        if len(idx) == 0:
            raise ValueError('The sampled training subset is empty.')
        model = MulticlassNeedMLP(need_bound=need_bound, random_state=index, max_iter=1000)
        model.fit(X_train[idx], h_train[idx])
        return model
    raise ValueError(f"unknown method {method!r}; known: {METHODS}")


def _derive(model, X_values, classes, need_threshold):
    if not np.array_equal(model.classes_, classes):
        raise ValueError('Unexpected class order or incomplete output support.')
    q = model.predict_proba(X_values)
    p = np.clip(q[:, classes >= need_threshold].sum(axis=1), 0.0, 1.0)
    return q, p


def _model_task(index, method, X_train, h_train, X_val, X_test,
                y_train, y_val, h_train_lab, h_val_lab, h_test_lab,
                need_bound, need_threshold):
    """Fit one model and return everything the output file needs."""
    classes = np.arange(need_bound + 1)
    model = build_model(method, X_train, h_train, index, need_bound)

    q_test, p_test = _derive(model, X_test, classes, need_threshold)
    q_train, p_train = _derive(model, X_train, classes, need_threshold)
    q_val, p_val = _derive(model, X_val, classes, need_threshold)

    # both collapses of the same distribution, so argmax costs no extra fit
    mu = {
        "average": q_test @ classes.astype(float),
        "argmax": q_test.argmax(axis=1),
    }
    tr_ce = log_loss(h_train_lab, q_train, labels=classes)
    va_ce = log_loss(h_val_lab, q_val, labels=classes)
    te_ce = log_loss(h_test_lab, q_test, labels=classes)
    return {
        "index": index,
        "p_test": p_test,
        "mu": mu,
        "training_loss": log_loss(y_train, p_train, labels=[0, 1]),
        "validation_loss": log_loss(y_val, p_val, labels=[0, 1]),
        "losses_row": {
            "training_cross_entropy": tr_ce,
            "validation_cross_entropy": va_ce,
            "test_cross_entropy": te_ce,
            # the notebook fills these with the cross-entropy values rather than
            # the binary losses above; kept so the file matches his contract
            "training_log_loss": tr_ce,
            "validation_log_loss": va_ce,
            "n_train_sample": int(len(X_train) * BOOTSTRAP_SIZE),
            "n_iter": model.n_iter_,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-threshold", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--method", default="bootstrap", choices=METHODS)
    ap.add_argument("--n-models", type=int, default=N_MODELS)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    need_threshold = args.need_threshold
    qualification_col = f"qualified_gagne_{need_threshold}"
    method = args.method

    fp_input = ROOT / "data/obermeyer/obermeyer_data_cleaned.csv"
    if not fp_input.exists():
        sys.exit(f"missing input: {fp_input}")

    def out_path(pred_method):
        return ROOT / ("data/predictions/"
                       f"test_{method}_{MODEL_CLASS}_{pred_method}_{qualification_col}"
                       f"{args.out_suffix}.csv")

    fp_losses = ROOT / ("data/predictions/losses/"
                        f"test_{method}_{MODEL_CLASS}_{qualification_col}{args.out_suffix}.csv")

    start = time.time()
    print(f"[start] method={method} threshold={need_threshold} "
          f"models={args.n_models} seeds={args.n_seeds} n_jobs={args.n_jobs}", flush=True)

    df = pd.read_csv(fp_input)
    need_bound = int(df[NEED_COL].max())
    classes = np.arange(need_bound + 1)
    X = df[FEATURES + OTHER_VARIABLES]
    h = pd.Series(validate_needs(df[NEED_COL], need_bound), index=df.index, name=NEED_COL)
    y = df[qualification_col]
    print(f"{len(df):,} people; need classes 0..{need_bound}", flush=True)

    per_method_output = {m: [] for m in PRED_METHODS}
    losses_rows = []

    with parallel_config(backend="loky", inner_max_num_threads=1):
        for seed in range(args.n_seeds):
            X_train, X_test, y_train, y_test, h_train, h_test = train_test_split(
                X, y, h, test_size=TEST_SIZE, random_state=seed)
            X_train, X_val, y_train, y_val, h_train, h_val = train_test_split(
                X_train, y_train, h_train, test_size=VALIDATION_SIZE, random_state=seed)

            cost = X_test['cost_t'].to_numpy()
            gagne = X_test['gagne_sum_t'].to_numpy()
            person_id = X_test['person_id'].to_numpy()

            X_train = X_train.drop(columns=OTHER_VARIABLES).to_numpy().copy()
            X_val = X_val.drop(columns=OTHER_VARIABLES).to_numpy().copy()
            X_test = X_test.drop(columns=OTHER_VARIABLES).to_numpy().copy()
            h_train = h_train.to_numpy().copy()
            h_val = h_val.to_numpy().copy()
            h_test = h_test.to_numpy().copy()
            y_train = y_train.to_numpy().copy()
            y_val = y_val.to_numpy().copy()
            y_test = y_test.to_numpy().copy()

            t0 = time.time()
            results = Parallel(n_jobs=args.n_jobs, batch_size=1, pre_dispatch="2*n_jobs")(
                delayed(_model_task)(
                    i, method, X_train, h_train, X_val, X_test,
                    y_train, y_val, h_train, h_val, h_test,
                    need_bound, need_threshold)
                for i in range(args.n_models))
            results.sort(key=lambda r: r["index"])   # loky returns out of order
            print(f"  seed {seed}: {args.n_models} models in {(time.time()-t0)/60:.1f} min",
                  flush=True)

            predictions, training_loss, validation_loss = {}, {}, {}
            expected_needs = {m: {} for m in PRED_METHODS}
            for r in results:
                name = f"m_{r['index'] + 1}"
                predictions[name] = r["p_test"]
                training_loss[name] = r["training_loss"]
                validation_loss[name] = r["validation_loss"]
                for m in PRED_METHODS:
                    expected_needs[m][f"exp_need_{name}"] = r["mu"][m]
                losses_rows.append({"seed": seed, "model": name, **r["losses_row"]})

            base = pd.concat([
                pd.DataFrame(predictions),
                pd.DataFrame(training_loss, index=[0]),
                pd.DataFrame(validation_loss, index=[0]),
            ]).reset_index(drop=True)
            base['y'] = np.concatenate([y_test, [np.nan, np.nan]])
            base['person_id'] = np.concatenate([person_id, [-2, -1]])
            base['cost_t'] = np.concatenate([cost, [np.nan, np.nan]])
            base['gagne_sum_t'] = np.concatenate([gagne, [np.nan, np.nan]])
            base['seed'] = seed

            for m in PRED_METHODS:
                need_df = pd.DataFrame(expected_needs[m]).reindex(base.index)
                per_method_output[m].append(pd.concat([base, need_df], axis=1))

    for m in PRED_METHODS:
        out = pd.concat(per_method_output[m])
        for c in out.columns:
            if c.startswith('m_'):
                out[c] = out[c].astype('float32')
        fp = out_path(m)
        fp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(fp, index=False)
        print(f"  wrote {fp.name}  shape={out.shape}", flush=True)

    fp_losses.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(losses_rows).to_csv(fp_losses, index=False)
    print(f"[done] threshold={need_threshold} in {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
