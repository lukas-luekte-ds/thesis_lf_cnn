import torch


def make_loss(cfg):
    lt = cfg["train"]["loss_type"].lower()

    if lt == "mae":
        def loss_fun(pred, y):
            return (pred - y).abs().mean()
        return loss_fun

    if lt == "mse":
        def loss_fun(pred, y):
            return torch.nn.functional.mse_loss(pred, y)
        return loss_fun

    if lt == "huber":
        def loss_fun(pred, y):
            delta = cfg["train"]["huber_delta"]
            return torch.nn.functional.huber_loss(pred, y, delta=delta)
        return loss_fun

    raise ValueError(f"Invalid loss type '{lt}'. Supported loss types are: mae, mse, huber.")

def make_optimizer(model, cfg):
    opt_name = cfg["train"]["optimizer"].lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                                weight_decay=float(cfg["train"]["weight_decay"]))
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    else:
        raise ValueError(f"Invalid Optimizer: {opt_name}")