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

from MHA import MLP
from utils import print_gpu_memory, check_model_on_gpu, set_seed


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

class ProtT5Classifier(nn.Module):
    """
    pT5 encoder + One Hidden Layer Perceptron
    Simple mean pool on residue embeddings
    """
    def __init__(self, num_classes, loss_weights=None):
        super().__init__()
        self.classifier = MLP(input_dim=1024, hidden_dim=512, output_dim=num_classes)

        if loss_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.as_tensor(loss_weights, device=DEVICE, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):
        logits = self.classifier(embeddings)

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
    print("compute_metrics called!")
    # ... rest of your code


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

def get_input_data():

    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .head(1000)
        .select(["protein_emb", "category"])
        .collect()
        .rename({"protein_emb": "embeddings", "category": "labels"})
        .to_pandas()
    )

    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(df["labels"].values), 
        y=df["labels"].values
    )

    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    training_dataset.set_format("torch", columns=["embeddings","labels"])

    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .head(1000)
        .select(["protein_emb", "category"])
        .collect()
        .rename({"protein_emb": "embeddings", "category": "labels"})
        .to_pandas()
    )

    eval_dataset = Dataset.from_pandas(df, preserve_index=False)
    eval_dataset.set_format("torch", columns=["embeddings","labels"])

    return training_dataset, eval_dataset, loss_weights

def main(): 

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory()
    print("     VRAM cleared!")

    # ========== Load / prepare data ==========
    
    train_ds, eval_ds, loss_weights = get_input_data()

    MODEL = ProtT5Classifier(num_classes = 5, loss_weights = loss_weights).to(DEVICE)

    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(
        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/MLP",
        num_train_epochs=100,
        per_device_train_batch_size=126,
        per_device_eval_batch_size=126,
        eval_strategy="epoch",
        fp16=False,
        save_strategy="epoch",
        logging_steps=100,
        report_to=["wandb"],
        run_name="MLP",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,        
        compute_metrics=compute_metrics  
    )

    wandb.init(project="MLP", name="Simple MLP")

    trainer.train()


if __name__ == "__main__": 

    main()