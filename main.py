import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessor import process_file
import lstm as lstm_module
import rnn as rnn_module
import lr as lr_module
import xgb as xgb_module

SEQ_LEN = 180
HORIZON_DAYS = 182
MIN_TRAIN_EXAMPLES = 50
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


def aggregate_window_features(X):
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    return np.concatenate([last, mean, std, mn, mx], axis=1)


def run_lstm(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            prep = process_file(
                csv_path,
                seq_len=SEQ_LEN,
                horizon_days=HORIZON_DAYS,
                min_train_examples=MIN_TRAIN_EXAMPLES
            )
            if prep is None:
                print(f"Skipping {csv_path}")
                continue

            ticker = prep["ticker"]
            X_train = prep["X_train"]
            y_train = prep["y_train"]
            X_test = prep["X_test"]
            y_test = prep["y_test"]

            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32)

            n_val = max(1, int(0.1 * len(X_tensor)))
            n_train = len(X_tensor) - n_val

            dataset = TensorDataset(X_tensor, y_tensor)
            train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            n_features = X_train.shape[2]
            model = lstm_module.build_model(
                n_features=n_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            )

            out = lstm_module.train(
                model,
                train_loader,
                val_loader,
                device=device,
                epochs=EPOCHS,
                lr=LR_RATE,
                patience=PATIENCE
            )
            model = out["model"]

            mse_test = None
            if X_test is not None and X_test.shape[0] > 0:
                test_ds = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32)
                )
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

                eval_out = lstm_module.evaluate(model, test_loader, device)
                mse_test = eval_out["mse"]

                pred_df = pd.DataFrame({
                    "ticker": ticker,
                    "target_date": prep["test_dates"],
                    "y_true": eval_out["trues"].flatten(),
                    "y_pred": eval_out["preds"].flatten()
                })
                pred_path = os.path.join(out_dir, f"{ticker}_lstm_predictions.csv")
                pred_df.to_csv(pred_path, index=False)

            
            model_path = os.path.join(out_dir, f"{ticker}_lstm.pt")
            lstm_module.save(model, model_path, metadata={"feature_cols": prep["feature_cols"], "seq_len": SEQ_LEN})

            results.append({
                "ticker": ticker,
                "train_examples": X_train.shape[0],
                "test_examples": 0 if X_test is None else X_test.shape[0],
                "mse": mse_test,
                "model_path": model_path
            })

            print(f"[LSTM] Finished {ticker}")

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "summary_lstm.csv"), index=False)


def run_rnn(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        try:
            prep = process_file(csv_path, seq_len=SEQ_LEN, horizon_days=HORIZON_DAYS)
            if prep is None:
                continue

            ticker = prep["ticker"]
            X_train, y_train = prep["X_train"], prep["y_train"]
            X_test, y_test = prep["X_test"], prep["y_test"]

            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32)

            n_val = max(1, int(0.1 * len(X_tensor)))
            n_train = len(X_tensor) - n_val

            dataset = TensorDataset(X_tensor, y_tensor)
            train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            n_features = X_train.shape[2]
            model = rnn_module.VanillaRNN(
                n_features=n_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            )

            out = rnn_module.train(
                model, train_loader, val_loader,
                device=device, epochs=EPOCHS, lr=LR_RATE, patience=PATIENCE
            )
            model = out["model"]
            mse_test = None
            if X_test is not None and X_test.shape[0] > 0:
                test_ds = TensorDataset(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32)
                )
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
                eval_out = rnn_module.evaluate(model, test_loader, device)
                mse_test = eval_out["mse"]

            model_path = os.path.join(out_dir, f"{ticker}_rnn.pt")
            rnn_module.save_state(model_path, model, metadata={"feature_cols": prep["feature_cols"], "seq_len": SEQ_LEN})

            results.append({
                "ticker": ticker,
                "train_examples": X_train.shape[0],
                "test_examples": 0 if X_test is None else X_test.shape[0],
                "mse": mse_test,
                "model_path": model_path
            })

            print(f"[RNN] Finished {ticker}")

        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "summary_rnn.csv"), index=False)


def run_lr(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        prep = process_file(csv_path, seq_len=SEQ_LEN, horizon_days=HORIZON_DAYS)
        if prep is None:
            continue

        ticker = prep["ticker"]
        X_train = aggregate_window_features(prep["X_train"])
        X_test = None if prep["X_test"] is None else aggregate_window_features(prep["X_test"])

        y_train, y_test = prep["y_train"], prep["y_test"]

        model = lr_module.train_lr(X_train, y_train)
        preds = None
        mse_test = None

        if X_test is not None and X_test.shape[0] > 0:
            preds = lr_module.predict_lr(model, X_test)
            mse_test = float(np.mean((preds - y_test)**2))

            pred_df = pd.DataFrame({
                "ticker": ticker,
                "target_date": prep["test_dates"],
                "y_true": y_test.flatten(),
                "y_pred": preds.flatten()
            })
            pred_df.to_csv(os.path.join(out_dir, f"{ticker}_lr_predictions.csv"), index=False)

        model_path = os.path.join(out_dir, f"{ticker}_lr.pkl")
        lr_module.save_lr(model_path, model, metadata={"feature_cols": prep["feature_cols"]})

        results.append({
            "ticker": ticker,
            "train_examples": X_train.shape[0],
            "test_examples": 0 if X_test is None else X_test.shape[0],
            "mse": mse_test,
            "model_path": model_path
        })

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "summary_lr.csv"), index=False)


def run_xgb(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for csv_path in glob.glob(os.path.join(data_dir, "*.csv")):
        prep = process_file(csv_path, seq_len=SEQ_LEN, horizon_days=HORIZON_DAYS)
        if prep is None:
            continue

        ticker = prep["ticker"]

        X_train = aggregate_window_features(prep["X_train"])
        X_test = None if prep["X_test"] is None else aggregate_window_features(prep["X_test"])

        y_train, y_test = prep["y_train"], prep["y_test"]

        model = xgb_module.train_xgb(X_train, y_train, params=XGB_PARAMS)

        mse_test = None
        if X_test is not None:
            preds = xgb_module.predict_xgb(model, X_test)
            mse_test = float(np.mean((preds - y_test)**2))

            pred_df = pd.DataFrame({
                "ticker": ticker,
                "target_date": prep["test_dates"],
                "y_true": y_test.flatten(),
                "y_pred": preds.flatten()
            })
            pred_df.to_csv(os.path.join(out_dir, f"{ticker}_xgb_predictions.csv"), index=False)

        model_path = os.path.join(out_dir, f"{ticker}_xgb.pkl")
        xgb_module.save_xgb(model_path, model, metadata={"feature_cols": prep["feature_cols"]})

        results.append({
            "ticker": ticker,
            "train_examples": X_train.shape[0],
            "test_examples": 0 if X_test is None else X_test.shape[0],
            "mse": mse_test,
            "model_path": model_path
        })

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "summary_xgb.csv"), index=False)



if __name__ == "__main__":
    data_directory = "data/csv_files"
    output_directory = "model_output"

    run_lstm(data_directory, os.path.join(output_directory, "lstm"))
    run_rnn(data_directory, os.path.join(output_directory, "rnn"))
    run_lr(data_directory, os.path.join(output_directory, "lr"))
    run_xgb(data_directory, os.path.join(output_directory, "xgb"))