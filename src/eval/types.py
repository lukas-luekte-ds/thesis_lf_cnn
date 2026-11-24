from dataclasses import dataclass

import numpy as np


# having one prediction frame guarantees the format of the predictions to be compatible and it contains all the arrays needed
@dataclass
class PredFrame:
    ids: np.ndarray                 # (N,)
    y_pred: np.ndarray              # (N, D)
    y_true: np.ndarray | None    # (N, D) or None
    eval_mask: np.ndarray | None # (N, D) bool or None