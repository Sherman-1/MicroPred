# ========== Local imports ==========
from pathlib import Path
import sys, os

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if parent.name == "MicroPred":
        repo_root = parent
        break
else:
    raise RuntimeError("Repository root 'MicroPred' not found.")

src_path = repo_root / "src"
exp_path = repo_root / "experiment"
sys.path.append(str(src_path))
sys.path.append(str(exp_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

from MHA import INT_TO_CLASS, CLASSES, CLASS_TO_INT

import torch 
import numpy as np
import polars as pl 


def compute_class_predictions(probabilities, threshold):
        predictions = []
        for proba_row in probabilities:
            multiple_classes = torch.sum(proba_row > 0.5)
            high_confidence_class = np.where(proba_row > threshold)[0]

            if multiple_classes > 1:
                predictions.append("Multiple")
            elif len(high_confidence_class) == 1:
                predictions.append(CLASSES[high_confidence_class[0]])
            else:
                predictions.append("Null")
        return predictions


df = pl.read_parquet("logits_dataframe.parquet")

