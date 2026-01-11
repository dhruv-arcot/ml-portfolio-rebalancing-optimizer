from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from typing import Dict, Any

def train_lr(X_train: np.ndarray, y_train: np.ndarray, **kwargs):
    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train.ravel())
    return model

def predict_lr(model, X: np.ndarray):
    return model.predict(X).reshape(-1, 1)

def save_lr(path: str, model, metadata: Dict[str, Any]):
    joblib.dump({'model': model, 'metadata': metadata}, path)
