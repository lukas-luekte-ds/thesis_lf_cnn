import torch
import numpy as np

# This is to evaluate the results 
# this should be in eval.py or utils.py later and maybe implement different eval strategies and encapsulate them as eval(mae, ...)
@torch.no_grad()
def eval_mae(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:   
        x = x.to(device).float()
        y = y.to(device).float()
        p = model(x)
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    P = np.concatenate(preds, 0)
    Y = np.concatenate(trues, 0)
    return float(np.mean(np.abs(P - Y)))

# maybe this can be somewhere else
@torch.no_grad()
def eval_mae_per_target(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _ in loader:
        x = x.to(device).float() 
        y = y.to(device).float()
        preds.append(model(x).cpu().numpy())
        trues.append(y.cpu().numpy())
    P = np.concatenate(preds, 0)
    Y = np.concatenate(trues, 0)
    per_target = np.mean(np.abs(P - Y), axis=0)  # shape (6,)
    return per_target.tolist()

