import joblib
from typing import Dict, Any
import numpy as np

try:
    import xgboost as xgb
except Exception as e:
    xgb = None

def train_xgb(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any] = None, num_boost_round: int = 200):
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    params = params or {'objective': 'reg:squarederror', 'max_depth':6, 'learning_rate':0.05, 'n_estimators':num_boost_round}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train.ravel(), verbose=False)
    return model

def predict_xgb(model, X: np.ndarray):
    return model.predict(X).reshape(-1, 1)

def save_xgb(path: str, model, metadata: Dict[str, Any]):
    joblib.dump({'model': model, 'metadata': metadata}, path)