"""Trains forward-NAV-return forecasters for each fund in the universe.

This is a thin adapter, not a rewrite: it reuses ../preprocessor.py's
process_file() and the four model modules (lstm/rnn/lr/xgb) exactly as
main.py does for US stocks, just pointed at india_mf_poc/data/nav/ and with
an India-appropriate backtest window (train 2020-2022, backtest from 2023
through whatever NAV history is latest -- no fixed 2025 cutoff).
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

import config

sys.path.insert(0, os.path.dirname(config.HERE))  # repo root, for the shared ML modules

# xgboost must be imported before torch: on macOS the two ship conflicting
# bundled OpenMP runtimes and importing torch first reliably segfaults the
# process the moment xgboost's C++ core is touched (confirmed locally). This
# also lurks in ../main.py, which imports torch before xgb -- it just never
# surfaced there because xgboost wasn't installed in this environment.
import xgb as xgb_module  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import TensorDataset, DataLoader  # noqa: E402

from preprocessor import process_file  # noqa: E402
import lstm as lstm_module  # noqa: E402

BATCH_SIZE = 64
EPOCHS = 50
LR_RATE = 1e-3
PATIENCE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
}


def _aggregate_window_features(X: np.ndarray) -> np.ndarray:
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([last, mean, std, mn, mx], axis=1)


def _process(label: str) -> dict:
    csv_path = os.path.join(config.NAV_DIR, f"{label}.csv")
    prep = process_file(
        csv_path,
        seq_len=config.SEQ_LEN,
        horizon_days=config.HORIZON_DAYS,
        min_target_date=pd.Timestamp(config.TRAIN_START),
        max_target_date=pd.Timestamp(config.BACKTEST_END),
        min_train_examples=config.MIN_TRAIN_EXAMPLES,
    )
    if prep is None:
        raise ValueError(f"process_file returned None for {label} -- not enough history/features")
    return prep


def train_xgb_forecaster(label: str, out_dir: str = config.MODEL_OUTPUT_DIR) -> dict:
    """Trains an XGBoost forecaster for one fund's forward NAV, returns test
    predictions keyed by target_date alongside the fitted model."""
    os.makedirs(out_dir, exist_ok=True)
    prep = _process(label)

    X_train = _aggregate_window_features(prep["X_train"])
    y_train = prep["y_train"]

    model = xgb_module.train_xgb(X_train, y_train, params=XGB_PARAMS)

    preds_by_date = {}
    if prep["X_test"] is not None and prep["X_test"].shape[0] > 0:
        X_test = _aggregate_window_features(prep["X_test"])
        preds = xgb_module.predict_xgb(model, X_test).flatten()
        for d, p in zip(prep["test_dates"], preds):
            preds_by_date[pd.Timestamp(d)] = float(p)

        mse = float(np.mean((preds - prep["y_test"].flatten()) ** 2))
        pd.DataFrame({
            "target_date": prep["test_dates"],
            "y_true": prep["y_test"].flatten(),
            "y_pred": preds,
        }).to_csv(os.path.join(out_dir, f"{label}_xgb_predictions.csv"), index=False)
        print(f"[XGB] {label}: train={X_train.shape[0]} test={X_test.shape[0]} mse={mse:.4f}")

    model_path = os.path.join(out_dir, f"{label}_xgb.pkl")
    xgb_module.save_xgb(model_path, model, metadata={"feature_cols": prep["feature_cols"]})

    return {
        "label": label,
        "model_path": model_path,
        "preds_by_date": preds_by_date,
        "last_close": float(prep["y_train"][-1][0]) if prep["y_train"].shape[0] else None,
    }


def train_lstm_forecaster(label: str, out_dir: str = config.MODEL_OUTPUT_DIR) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    prep = _process(label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = prep["X_train"], prep["y_train"]

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    n_val = max(1, int(0.1 * len(X_tensor)))
    n_train = len(X_tensor) - n_val
    dataset = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_features = X_train.shape[2]
    model = lstm_module.build_model(n_features=n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    out = lstm_module.train(model, train_loader, val_loader, device=device, epochs=EPOCHS, lr=LR_RATE, patience=PATIENCE)
    model = out["model"]

    preds_by_date = {}
    if prep["X_test"] is not None and prep["X_test"].shape[0] > 0:
        test_ds = TensorDataset(
            torch.tensor(prep["X_test"], dtype=torch.float32),
            torch.tensor(prep["y_test"], dtype=torch.float32),
        )
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        eval_out = lstm_module.evaluate(model, test_loader, device)
        for d, p in zip(prep["test_dates"], eval_out["preds"].flatten()):
            preds_by_date[pd.Timestamp(d)] = float(p)

        pd.DataFrame({
            "target_date": prep["test_dates"],
            "y_true": eval_out["trues"].flatten(),
            "y_pred": eval_out["preds"].flatten(),
        }).to_csv(os.path.join(out_dir, f"{label}_lstm_predictions.csv"), index=False)
        print(f"[LSTM] {label}: train={X_train.shape[0]} test={prep['X_test'].shape[0]} mse={eval_out['mse']:.4f}")

    model_path = os.path.join(out_dir, f"{label}_lstm.pt")
    lstm_module.save(model, model_path, metadata={"feature_cols": prep["feature_cols"], "seq_len": config.SEQ_LEN})

    return {
        "label": label,
        "model_path": model_path,
        "preds_by_date": preds_by_date,
        "last_close": float(prep["y_train"][-1][0]) if prep["y_train"].shape[0] else None,
    }


def train_universe(model: str = "xgb") -> dict:
    """Trains the given model type ('xgb' or 'lstm') for every fund in the
    universe. Returns {label: result_dict} where result_dict['preds_by_date']
    maps target_date -> predicted NAV, used by the rebalancer to rank funds."""
    train_fn = {"xgb": train_xgb_forecaster, "lstm": train_lstm_forecaster}[model]
    results = {}
    for fund in config.FUND_UNIVERSE:
        label = fund["label"]
        print(f"=== Training {model} forecaster for {label} ===")
        results[label] = train_fn(label)
    return results


if __name__ == "__main__":
    train_universe("xgb")
