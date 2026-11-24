import numpy as np
import torch


# This is to evaluate the results 
@torch.no_grad()
def eval_mae(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:   
        x = x.to(device).float()
        y = y.to(device).float()
        p = model(x)
        if p.shape != y.shape:
            raise ValueError(
                f"eval_mae shape mismatch: pred.shape={tuple(p.shape)}, y.shape={tuple(y.shape)}"
            )

        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    P = np.concatenate(preds, 0)
    Y = np.concatenate(trues, 0)
    return float(np.mean(np.abs(P - Y)))

@torch.no_grad()
def eval_mae_per_target(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:
        x = x.to(device).float() 
        y = y.to(device).float()
        p = model(x)
        if p.shape != y.shape:
            raise ValueError(
                f"eval_mae_per_target shape mismatch: pred.shape={tuple(p.shape)}, y.shape={tuple(y.shape)}"
            )
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    P = np.concatenate(preds, 0)
    Y = np.concatenate(trues, 0)
    per_target = np.mean(np.abs(P - Y), axis=0)  # shape (6,)
    return per_target.tolist()