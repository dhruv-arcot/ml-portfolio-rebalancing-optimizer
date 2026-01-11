from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import os

class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           
        out = out[:, -1, :]             
        out = self.bn(out)              
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out

def build_model(n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2) -> LSTMModel:
    """Return an uninitialized LSTMModel instance."""
    return LSTMModel(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

def _device_or_default(device: Optional[torch.device]) -> torch.device:
    return device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def train(model: nn.Module,
          train_loader,
          val_loader=None,
          device: Optional[torch.device]=None,
          epochs: int = 50,
          lr: float = 1e-3,
          ) -> Dict[str, Any]:
    
    
    device = _device_or_default(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    wait = 0

    history = {'train_loss': [], 'val_loss': []}
    best_epoch = None

    for ep in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = xb.size(0)
            train_sum += float(loss.item()) * b
            train_count += b
        train_loss = train_sum / max(1, train_count)
        history['train_loss'].append(train_loss)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    b = xb.size(0)
                    val_sum += float(loss.item()) * b
                    val_count += b
            val_loss = val_sum / max(1, val_count)
            history['val_loss'].append(val_loss)
        else:
            history['val_loss'].append(None)

        print(f"Epoch {ep:03d} | train_loss={train_loss:.6f} | val_loss={val_loss if val_loss is not None else 'N/A'}")

        if val_loss is not None:
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep
                wait = 0
            else:
                wait += 1

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return {"model": model, "history": history, "best_epoch": best_epoch, "best_val_loss": (best_val if best_state is not None else None)}

def evaluate(model: nn.Module, loader, device: Optional[torch.device] = None) -> Optional[Dict[str, Any]]:
    device = _device_or_default(device)
    model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            trues.append(yb.numpy())
    if len(preds) == 0:
        return None
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mse = float(mean_squared_error(trues, preds))
    return {"mse": mse, "preds": preds, "trues": trues}

def save(model: nn.Module, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)

def load(path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    device = _device_or_default(device)
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata", {})
    return state_dict, metadata
