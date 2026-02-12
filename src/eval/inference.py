import torch
import numpy as np
from .types import PredFrame
from typing import List

# layer 1a: Inference
@torch.no_grad()
def infer_val(model, dataloader, device) -> PredFrame:
    """
    generates the predictions for the validation and returns them as PredFrame
    """
    model.eval()
    model.to(device)

    ids_all ,pred_all, true_all, mask_all = [], [], [], []
    row_cursor = 0

    for x, y, flag in dataloader:
        B = int(x.shape[0])
        ids_all.append(np.arange(row_cursor, row_cursor + B, dtype=np.int64))
        row_cursor += B

        x = x.to(device).float()
        y_pred = model(x).detach().cpu().numpy()
        pred_all.append(y_pred)

        y_np = y.detach().cpu().numpy()
        true_all.append(y_np)

        m = flag.detach().cpu().numpy().astype(bool)
        mask_all.append(m)

    ids = np.concatenate(ids_all, axis=0)
    y_pred = np.concatenate(pred_all, axis=0)
    y_true = np.concatenate(true_all, axis=0)
    eval_mask = np.concatenate(mask_all, axis=0)

    if ids.shape[0] != y_pred.shape[0]:
        raise ValueError(f"N mismatch: ids={ids.shape[0]} vs y_pred={y_pred.shape[0]}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} != y_pred {y_pred.shape}")
    if eval_mask.shape != y_pred.shape:
        raise ValueError(f"mask {eval_mask.shape} != y_pred {y_pred.shape}")

    return PredFrame(ids=ids, y_pred=y_pred, y_true=y_true, eval_mask=eval_mask)

# layer 1b
@torch.no_grad()
def infer_test(model, dataloader, device) -> PredFrame:
    """
    generates the predictions for the testset with the given model and returns them as PredFrame
    """
    model.eval()
    model.to(device)

    ids_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []

    for ID, x in dataloader:
        if torch.is_tensor(ID):
            ID = ID.detach().cpu().numpy()
        ids_all.append(np.asarray(ID).reshape(-1))

        x = x.to(device).float()
        y_pred = model(x).detach().cpu().numpy()
        pred_all.append(y_pred)

    ids = np.concatenate(ids_all, axis=0)
    y_pred = np.concatenate(pred_all, axis=0)

    if ids.shape[0] != y_pred.shape[0]:
        raise ValueError(f"N mismatch: ids={ids.shape[0]} vs y_pred={y_pred.shape[0]}")

    return PredFrame(ids=ids, y_pred=y_pred, y_true=None, eval_mask=None)


#layer 2b
def subset_preds_by_ids(preds: PredFrame, ids_wanted: np.ndarray) -> PredFrame:
    """
    selects the wanted ids in the array and returns the result als PredFrame
    """
    ids_all = preds.ids.astype(np.int64, copy=False)
    ids_wanted = np.asarray(ids_wanted).astype(np.int64, copy=False)

    # sort once for fast lookup
    order = np.argsort(ids_all)
    ids_sorted = ids_all[order]

    pos = np.searchsorted(ids_sorted, ids_wanted)

    # validate found
    ok = (pos >= 0) & (pos < ids_sorted.size) & (ids_sorted[pos] == ids_wanted)
    if not np.all(ok):
        missing = ids_wanted[~ok][:10]
        raise ValueError(f"Some IDs not found in global test predictions. Example missing IDs: {missing}")

    idx = order[pos]  # indices into preds arrays
    return PredFrame(ids=ids_all[idx], y_pred=preds.y_pred[idx], y_true=None, eval_mask=None)

