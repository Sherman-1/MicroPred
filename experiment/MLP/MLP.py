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

from transformers import (

    TrainingArguments,
    Trainer
)

from datasets import Dataset
import wandb

import evaluate

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
    """
    def __init__(self, num_classes, loss_weights=None):
        super().__init__()
        self.classifier = MLP(input_dim=1024, hidden_dim=512, output_dim=num_classes)

        if loss_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                weight=torch.as_tensor(loss_weights, device=DEVICE, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, embeddings, labels=None):
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


#####################################
# EVALUATION METRICS
#####################################

accuracy_metric = evaluate.load("accuracy", zero_division=0)
f1_metric = evaluate.load("f1", zero_division=0)
precision_metric = evaluate.load("precision", zero_division=0)
recall_metric = evaluate.load("recall", zero_division=0)

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    if hasattr(logits, "cpu"):
        logits = logits.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    preds = np.argmax(logits, axis=-1)
    true_labels = np.argmax(labels, axis=-1) # Turn one-hot to class index

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

def get_input_data():

    print("Collecting training data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["protein_emb", "category", "one_hot"])
        .rename({"protein_emb": "embeddings", "one_hot": "labels"})
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
        .select(["protein_emb", "category", "one_hot"])
        .rename({"protein_emb": "embeddings", "one_hot": "labels"})
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
    
    train_ds, eval_ds, loss_weights = get_input_data()

    MODEL = ProtT5Classifier(num_classes = 5, loss_weights = loss_weights).to(DEVICE)

    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/MLP",
        num_train_epochs=100,
        
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
        run_name="F1_DeepSpeed",

        logging_strategy="no"
        
    )


    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,      
        compute_metrics=compute_metrics  
    )

    wandb.init(project="MLP", name="MLP_F1_DeepSpeed")

    trainer.train()


if __name__ == "__main__": 

    main()