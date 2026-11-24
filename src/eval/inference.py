import numpy as np
import torch

from .types import PredFrame


# layer 1a: Inference, unused since the validation uses the inference from training,
# could be used to validate an already trained model
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

    ids_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []

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

