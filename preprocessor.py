
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Close' not in df.columns:
        if 'Price' in df.columns:
            df['Close'] = df['Price']
        else:
            raise ValueError("DataFrame must contain 'Close' or 'Price' column")

    df['Close'] = df['Close'].astype(float)
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_5'] = df['Close'].pct_change(5)
    # use log1p for stability
    df['logret_1'] = np.log1p(df['ret_1'])
    # rolling means
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_21'] = df['Close'].rolling(window=21).mean()
    df['ma_63'] = df['Close'].rolling(window=63).mean()
    # volatility
    df['vol_21'] = df['logret_1'].rolling(window=21).std()
    df['vol_63'] = df['logret_1'].rolling(window=63).std()
    # momentum
    df['mom_21'] = df['Close'] / df['Close'].shift(21) - 1
    # range / ATR-like
    df['range'] = (df['High'] - df['Low']) / (df['Open'] + 1e-9)
    df['range_21'] = df['range'].rolling(window=21).mean()
    # volume features
    df['vol_mean_21'] = df['Volume'].rolling(window=21).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_mean_21'] + 1e-9)
    # cyclical date features
    df['dow'] = df['Date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month'] = df['Date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df


def build_supervised(df: pd.DataFrame,
                     feature_cols: List[str],
                     target_col: str = 'Close',
                     seq_len: int = 180,
                     horizon_days: int = 182,
                     min_target_date: pd.Timestamp = pd.Timestamp('2015-06-01'),
                     max_target_date: pd.Timestamp = pd.Timestamp('2025-12-31')
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_target_date = pd.to_datetime(min_target_date).normalize()
    max_target_date = pd.to_datetime(max_target_date).normalize()

    if 'Date' not in df.columns:
        raise ValueError("DataFrame must contain 'Date' column")

    df_local = df.copy()
    df_local['Date'] = pd.to_datetime(df_local['Date']).dt.normalize()
    df_local = df_local.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    if df_local.shape[0] == 0:
        empty_X = np.zeros((0, seq_len, len(feature_cols)))
        empty_y = np.zeros((0, 1))
        return empty_X, empty_y, np.array([]), np.array([])

    df_idx = df_local.set_index('Date', drop=False)
    idx = df_idx.index

    candidates = df_local.loc[(df_local['Date'] >= min_target_date) & (df_local['Date'] <= max_target_date), 'Date']
    if len(candidates) == 0:
        empty_X = np.zeros((0, seq_len, len(feature_cols)))
        empty_y = np.zeros((0, 1))
        return empty_X, empty_y, np.array([]), np.array([])

    xs = []
    ys = []
    tdates = []
    iends = []

    for tdate in candidates:
        tdate = pd.to_datetime(tdate).normalize()
        desired_input_end = (tdate - pd.Timedelta(days=int(horizon_days))).normalize()
       
        pos = idx.searchsorted(desired_input_end, side='right') - 1
        if pos < 0:
       
            continue
        actual_input_end = idx[pos]
        start_pos = pos - (seq_len - 1)
        if start_pos < 0:
       
            continue
        window = df_idx.iloc[start_pos: pos + 1]
        if len(window) != seq_len:
            continue

       
        tpos = idx.get_indexer([tdate])[0]
        if tpos == -1:
            matches = np.where(idx == tdate)[0]
            if matches.size == 0:
                continue
            tdate_used = idx[matches[-1]]
        else:
            tdate_used = idx[tpos]

        X_window = window[feature_cols].values
        y_value = df_idx.at[tdate_used, target_col]
        if np.isnan(X_window).any() or pd.isna(y_value):
            continue

        xs.append(X_window)
        ys.append(y_value)
        tdates.append(tdate_used)
        iends.append(actual_input_end)

    if len(xs) == 0:
        empty_X = np.zeros((0, seq_len, len(feature_cols)))
        empty_y = np.zeros((0, 1))
        return empty_X, empty_y, np.array([]), np.array([])

    X = np.array(xs)
    y = np.array(ys).reshape(-1, 1)
    return X, y, np.array(tdates), np.array(iends)

def process_file(csv_path: str,
                 seq_len: int = 180,
                 horizon_days: int = 182,
                 min_target_date: Optional[pd.Timestamp] = None,
                 max_target_date: Optional[pd.Timestamp] = None,
                 min_train_examples: int = 50
                 ) -> Optional[Dict[str, Any]]:
    if min_target_date is None:
        min_target_date = pd.Timestamp('2015-06-01')
    if max_target_date is None:
        max_target_date = pd.Timestamp('2025-12-31')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if 'Date' not in df.columns:

        return None


    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)


    if 'Close' not in df.columns:
        if 'Price' in df.columns:
            df['Close'] = df['Price']
        else:
            return None

    # create features
    df_feat = create_features(df)
    # drop initial NaNs generated by rolling windows
    df_feat = df_feat.dropna().reset_index(drop=True)
    if df_feat.shape[0] == 0:
        return None

    exclude = {'Date', 'Price', 'Close'}
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    if len(feature_cols) == 0:
        return None

    
    X, y, target_dates, input_end_dates = build_supervised(df_feat, feature_cols,
                                                           target_col='Close',
                                                           seq_len=seq_len,
                                                           horizon_days=horizon_days,
                                                           min_target_date=min_target_date,
                                                           max_target_date=max_target_date)
    if X is None or X.shape[0] == 0:
        return None

    
    train_mask = (target_dates >= pd.Timestamp('2020-01-01')) & (target_dates <= pd.Timestamp('2022-12-31'))
    test_mask = (target_dates >= pd.Timestamp('2023-01-01')) & (target_dates <= pd.Timestamp('2025-12-31'))

    X_train = X[train_mask]; y_train = y[train_mask]
    X_test = X[test_mask]; y_test = y[test_mask]
    train_dates = target_dates[train_mask]; test_dates = target_dates[test_mask]

    n_train = 0 if X_train is None else int(X_train.shape[0])
    n_test = 0 if X_test is None else int(X_test.shape[0])
    ticker = os.path.splitext(os.path.basename(csv_path))[0]

    if n_train < min_train_examples:
        return None

    
    n_examples, seq_l, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape[0], seq_l, n_features)
    X_test_scaled = None
    if n_test > 0:
        X_test_flat = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], seq_l, n_features)

    return {
        "ticker": ticker,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "feature_cols": feature_cols,
        "scaler": scaler
    }
