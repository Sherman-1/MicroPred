#!/usr/bin/env python3

"""
======================================================
Author: Simon HERMAN
Github: https://github.com/Sherman-1
======================================================
"""

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (

    TrainingArguments,
    Trainer
)

from datasets import Dataset
import wandb

# ========== Local imports ==========
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_path = root_dir / 'src'
sys.path.append(str(src_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

from MHA import MLP, MHAPooling
from utils import print_gpu_memory, check_model_on_gpu, set_seed

# ===================================

#####################################
# GLOBALS / CONFIG
#####################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    sys.exit("Cuda-compatible GPU not found!")

set_seed(66)

#####################################
# MODEL DEFINITION
#####################################

class MHAP_MLP(nn.Module):
    """
    forward batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> mhap_mlp = MHAP_MLP(input_embed_dim = 1024, output_embed_dim = 128, num_classes = 5)
    >>> mlp_output = mhap_mlp.forward(batch, mask = None)
    >>> mlp_output.shape
    torch.Size([64, 5])
    """
    def __init__(self, 
                num_classes:int, 
                input_embed_dim:int, 
                mlp_hidden_dim:int,
                output_embed_dim:int, 
                loss_weights:torch.Tensor
            ):
        
        super(MHAP_MLP, self).__init__()
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim)
        self.mlp = MLP(input_dim=output_embed_dim, output_dim=num_classes, hidden_dim = mlp_hidden_dim)

        if loss_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.as_tensor(loss_weights, device=DEVICE, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, attention_mask, labels):
        attn_output, _ = self.mhap(embeddings, attention_mask)  
        logits = self.mlp(attn_output)      
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits

#####################################
# EVALUATION METRICS
#####################################

def compute_metrics(eval_pred):

    """
    """
    logits, labels = eval_pred

    if not isinstance(logits, np.ndarray):
        logits = logits.numpy() if hasattr(logits, "numpy") else np.array(logits)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)

    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

#####################################
# DATA LOADING 
#####################################

def padding_collate(batch):
    """
    >>> import torch
    >>> import torch.nn.functional as F
    >>> _ = torch.manual_seed(0)
    >>> batch = [
    ...     {"embeddings": torch.rand(2,5), "labels": 0},
    ...     {"embeddings": torch.rand(4,5), "labels": 1}
    ... ]
    >>> out = padding_collate(batch)
    >>> out["embeddings"].shape
    torch.Size([2, 4, 5])
    >>> sum_vector = out["embeddings"][0][-2:].sum(dim=0).sum(dim=0).item()
    >>> print(sum_vector)
    0.0
    """
    embeddings = [item["embeddings"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    max_len = max(emb.shape[0] for emb in embeddings)
    
    padded_embeddings = []
    attention_masks = []
    
    for emb in embeddings:
        seq_len, emb_dim = emb.shape
        pad_len = max_len - seq_len
        if pad_len > 0:
            padded_emb = F.pad(emb, (0, 0, 0, pad_len))
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_emb = emb
            mask = torch.ones(seq_len, dtype=torch.bool)
            
        padded_embeddings.append(padded_emb)
        attention_masks.append(mask)
    
    batch_embeddings = torch.stack(padded_embeddings)
    batch_labels = torch.tensor(labels)
    batch_attention_mask = torch.stack(attention_masks)
    batch_attention_mask = ~batch_attention_mask 
    
    return {
        "embeddings": batch_embeddings,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask
    }


def get_input_data():

    print("Collecting training data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["residue_emb", "category"])
        .head(100)
        .collect()
        .rename({"residue_emb": "embeddings", "category": "labels"})
        .to_pandas()
    )

    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(df["labels"].values), 
        y=df["labels"].values
    )
    print(f"    Building dataset ... ")
    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    training_dataset.set_format("torch", columns=["embeddings","labels"])

    print(f"Collecting eval data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["residue_emb", "category"])
        .head(100)
        .collect()
        .rename({"residue_emb": "embeddings", "category": "labels"})
        .to_pandas()
    )
    print(f"    Building dataset ... ")
    eval_dataset = Dataset.from_pandas(df, preserve_index=False)
    eval_dataset.set_format("torch", columns=["embeddings","labels"])

    return training_dataset, eval_dataset, loss_weights

def main(): 

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory()
    print("     VRAM cleared!")

    # ========== Load / prepare data ==========

    print(f"Importing data ...")
    train_ds, eval_ds, loss_weights = get_input_data()

    print(f"Sending model to device ")
    MODEL = MHAP_MLP(
                    loss_weights = loss_weights,
                    input_embed_dim = 1024,
                    output_embed_dim = 256,
                    mlp_hidden_dim = 128,
                    num_classes = 5
                    ).to(DEVICE)

    from torch.utils.data import DataLoader

    dl = DataLoader(train_ds, collate_fn = padding_collate, batch_size = 10)

    batch = next(iter(dl)) 

    embed = batch["embeddings"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)

    MODEL.train()
    output = MODEL(embed, mask, labels)
    print(output)



if __name__ == "__main__":

    import argparse
    import doctest
    import sys

    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                func = getattr(sys.modules[__name__], f)   

                doctest.run_docstring_examples(
                    func,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )

        sys.exit()

    main()