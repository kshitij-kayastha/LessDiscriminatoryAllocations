"""Generate bootstrap prediction files, one per qualification column.

Faithful port of test/01-predictions_bootstrap.ipynb so the three targets can be
produced in one unattended run instead of editing TARGET_VARIABLE by hand.

    uv run python scripts/make_predictions.py --target qualified_gagne_1
    uv run python scripts/make_predictions.py --all
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

NUM_RANDOM_SEEDS = 10
NUM_BOOTSTRAP_MODELS = 100
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25
BOOTSTRAP_SIZE = 0.50

FEATURES = [
    'dem_female', 'dem_age_band_18-24_tm1', 'dem_age_band_25-34_tm1', 'dem_age_band_35-44_tm1',
    'dem_age_band_45-54_tm1', 'dem_age_band_55-64_tm1', 'dem_age_band_65-74_tm1', 'dem_age_band_75+_tm1',
    'hypertension_elixhauser_tm1', 'cost_dialysis_tm1', 'cost_emergency_tm1', 'cost_home_health_tm1',
    'cost_ip_medical_tm1', 'cost_ip_surgical_tm1', 'cost_laboratory_tm1', 'cost_op_primary_care_tm1',
    'cost_op_specialists_tm1', 'cost_op_surgery_tm1', 'cost_other_tm1', 'cost_pharmacy_tm1',
    'cost_physical_therapy_tm1', 'cost_radiology_tm1', 'gagne_sum_tm1',
]
OTHER_VARIABLES = ['person_id', 'gagne_sum_t', 'cost_t']

TARGETS = ["qualified_gagne_1", "qualified_gagne_2", "qualified_gagne_3"]


def get_bootstrap_model(X_train, y_train, random_seed, model_class, bootstrap_size=BOOTSTRAP_SIZE):
    np.random.seed(random_seed)
    idx = np.random.permutation(len(X_train))[:int(len(X_train) * bootstrap_size)]
    if model_class == "lr":
        model = LogisticRegression()
    elif model_class == "rf":
        model = RandomForestClassifier(random_state=random_seed)
    elif model_class == "nn":
        model = MLPClassifier(random_state=random_seed, max_iter=1000)
    else:
        raise ValueError(f"unknown model class {model_class}")
    model.fit(X_train[idx], y_train[idx])
    return model


def build(target, model_class, out_dir):
    fp_input = ROOT / "data/obermeyer/obermeyer_data_cleaned.csv"
    fp_output = out_dir / f"test_bootstrap_{model_class}_{target}.csv"

    df = pd.read_csv(fp_input)
    X = df[FEATURES + OTHER_VARIABLES]
    y = df[target]

    output = []
    for random_seed in range(NUM_RANDOM_SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=random_seed)

        cost = X_test["cost_t"].to_numpy()
        gagne = X_test["gagne_sum_t"].to_numpy()
        person_id = X_test["person_id"].to_numpy()

        X_train = X_train.drop(columns=OTHER_VARIABLES).to_numpy().copy()
        y_train = y_train.to_numpy().copy()
        X_val = X_val.drop(columns=OTHER_VARIABLES).to_numpy().copy()
        y_val = y_val.to_numpy().copy()
        X_test = X_test.drop(columns=OTHER_VARIABLES).to_numpy().copy()
        y_test = y_test.to_numpy().copy()

        predictions, training_loss, validation_loss = {}, {}, {}
        for i in tqdm(range(NUM_BOOTSTRAP_MODELS), desc=f"{target} seed {random_seed}"):
            model = get_bootstrap_model(X_train, y_train, random_seed=i, model_class=model_class)
            predictions[f"m_{i+1}"] = model.predict_proba(X_test)[:, 1]
            training_loss[f"m_{i+1}"] = log_loss(y_train, model.predict_proba(X_train))
            validation_loss[f"m_{i+1}"] = log_loss(y_val, model.predict_proba(X_val))

        predictions_df = pd.concat([
            pd.DataFrame(predictions),
            pd.DataFrame(training_loss, index=[0]),
            pd.DataFrame(validation_loss, index=[0]),
        ]).reset_index(drop=True)

        # person_id -1 marks the validation-loss row, -2 the training-loss row.
        predictions_df["y"] = np.concatenate([y_test, [np.nan, np.nan]])
        predictions_df["person_id"] = np.concatenate([person_id, [-2, -1]])
        predictions_df["cost_t"] = np.concatenate([cost, [np.nan, np.nan]])
        predictions_df["gagne_sum_t"] = np.concatenate([gagne, [np.nan, np.nan]])
        predictions_df["seed"] = random_seed

        output.append(predictions_df)

    output = pd.concat(output)
    for c in output.columns:
        if c.startswith("m_"):
            output[c] = output[c].astype("float32")

    fp_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(fp_output, index=False)
    return fp_output, output.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS)
    ap.add_argument("--all", action="store_true", help="build all three qualification columns")
    ap.add_argument("--model-class", default="nn", choices=["nn", "lr", "rf"])
    ap.add_argument("--out-dir", default=str(ROOT / "data/predictions"))
    args = ap.parse_args()

    targets = TARGETS if args.all else [args.target]
    if targets == [None]:
        ap.error("pass --target or --all")

    for target in targets:
        start = time.time()
        fp, shape = build(target, args.model_class, Path(args.out_dir))
        print(f"wrote {fp}  shape={shape}  ({time.time()-start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
