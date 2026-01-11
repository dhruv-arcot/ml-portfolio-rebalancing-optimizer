
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, Any

# ---------------------------
# Models
# ---------------------------
class RNN(nn.Module):
    def __init__(self, n_features: int, hidden_size: int =128, num_layers: int =2, dropout: float =0.2, nonlinearity: str='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, nonlinearity=nonlinearity, dropout=dropout if num_layers>1 else 0.0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def train(model: nn.Module,
          train_loader,
          val_loader,
          device: torch.device,
          epochs: int = 50,
          lr: float = 1e-3,
          patience: int = 8) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': []}

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b = xb.size(0)
            train_loss += float(loss.item()) * b
            n += b
        train_loss = train_loss / max(1, n)

        # validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            running = 0.0; total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    b = xb.size(0)
                    running += float(loss.item()) * b
                    total += b
            val_loss = running / max(1, total)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {ep:03d} train_loss={train_loss:.6f} val_loss={val_loss if val_loss is not None else 'N/A'}")

        if val_loss is not None:
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history}

def evaluate(model: nn.Module, loader, device: torch.device):
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
    mse = mean_squared_error(trues, preds)
    return {"mse": mse, "preds": preds, "trues": trues}

def save_state(path: str, model: nn.Module, metadata: dict):
    # saves state_dict + metadata dict
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
