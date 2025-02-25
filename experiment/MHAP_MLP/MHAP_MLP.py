#!/usr/bin/env python3

"""
======================================================
Author: Simon HERMAN
Github: https://github.com/Sherman-1
======================================================
"""

import warnings
import logging

# Fuck em logs
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import compute_class_weight

from transformers import (

    TrainingArguments,
    Trainer
)

import evaluate

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
                loss_weights:torch.Tensor = None
            ):
        
        super(MHAP_MLP, self).__init__()
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim)
        self.mlp = MLP(input_dim=output_embed_dim, output_dim=num_classes, hidden_dim = mlp_hidden_dim)

        if loss_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                weight=torch.as_tensor(loss_weights, device=DEVICE, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, embeddings, attention_mask, labels = None):
        attn_output, _ = self.mhap(embeddings, attention_mask)  
        logits = self.mlp(attn_output)      
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


#####################################
# EVALUATION METRICS
#####################################
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    # Convert logits and one-hot labels to numpy arrays (if they aren't already).
    if hasattr(logits, "cpu"):
        logits = logits.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    preds = np.argmax(logits, axis=-1)
    true_labels = np.argmax(labels, axis=-1)

    accuracy = accuracy_metric.compute(predictions=preds, references=true_labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=true_labels, average="macro")["f1"]
    precision = precision_metric.compute(predictions=preds, references=true_labels, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=true_labels, average="macro")["recall"]

    return {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "eval_precision": precision,
        "eval_recall": recall,
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
    labels = [torch.as_tensor(item["labels"]) for item in batch]  
    max_len = max(emb.shape[0] for emb in embeddings)

    padded_embeddings = []
    attention_masks = []

    for emb in embeddings:
        seq_len, emb_dim = emb.shape
        pad_len = max_len - seq_len
        if pad_len > 0:
            padded_emb = F.pad(emb, (0, 0, 0, pad_len))  # Pad only along sequence length
            mask = torch.cat([torch.ones(seq_len, dtype=torch.bool), 
                              torch.zeros(pad_len, dtype=torch.bool)])
        else:
            padded_emb = emb
            mask = torch.ones(seq_len, dtype=torch.bool)

        padded_embeddings.append(padded_emb)
        attention_masks.append(mask)

    batch_embeddings = torch.stack(padded_embeddings)
    batch_labels = torch.stack(labels)  
    batch_attention_mask = torch.stack(attention_masks)
    batch_attention_mask = ~batch_attention_mask  
    # Because of custom MultiHeadAttention implementation ! 
    # See these lines in the MHA definition :
    # if mask is not None:
    #        mask = mask.repeat(n_head, 1, 1) // (n*b) x .. x ..
    #    if key_padding_mask is not None:  //(sz_b, len_k)
    #        key_padding_mask = torch.stack([key_padding_mask,]*n_head, dim=0).reshape(sz_b*n_head, 1, len_k) * torch.ones(sz_b*n_head, len_q, 1, dtype=torch.bool, device = q.device)
    #        if mask is not None:
    #            mask = mask + key_padding_mask
    #        else:
    #            mask = key_padding_mask
    
    return {
        "embeddings": batch_embeddings,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask
    }

def get_input_data():

    print("Collecting training data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["residue_emb", "category", "one_hot"])
        .rename({"residue_emb": "embeddings", "one_hot": "labels"})
    )

    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(df.select("category").collect().to_series().to_list()), 
        y=df.select("category").collect().to_series().to_list()
    )
    print(f"    Building dataset ... ")
    training_dataset = Dataset.from_pandas(df.select(pl.exclude("category")).collect().to_pandas(), preserve_index=False)
    training_dataset.set_format("torch", columns=["embeddings","labels"])

    print(f"Collecting eval data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["residue_emb", "category", "one_hot"])
        .rename({"residue_emb": "embeddings", "one_hot": "labels"})
    )
    print(f"    Building dataset ... ")
    eval_dataset = Dataset.from_pandas(df.select(pl.exclude("category")).collect().to_pandas(), preserve_index=False)
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

    check_model_on_gpu(MODEL)

    
    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/MHAP_MLP",
        num_train_epochs=30,
        
        per_device_train_batch_size=512,
        per_device_eval_batch_size=1024,
        eval_strategy="epoch",
        remove_unused_columns=True,  
        
        fp16=False,
        deepspeed="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ds_config.json",
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",   
        greater_is_better=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to=["wandb"],
        run_name="F1_DeepSpeed"
        
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,      
        compute_metrics=compute_metrics,
        data_collator = padding_collate
    )
    wandb.init(project="MHAP_MLP", name="F1_DeepSpeed")

    trainer.train()


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