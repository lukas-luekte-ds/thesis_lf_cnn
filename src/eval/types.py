from dataclasses import dataclass
from typing import Optional
import numpy as np

# having one prediction frame guarantees the format of the predictions to be compatible and it contains all the arrays needed
@dataclass
class PredFrame:
    ids: np.ndarray                 # (N,)
    y_pred: np.ndarray              # (N, D)
    y_true: Optional[np.ndarray]    # (N, D) or None
    eval_mask: Optional[np.ndarray] # (N, D) bool or None