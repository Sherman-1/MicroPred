#!/usr/bin/env python3
"""
======================================================
Author: Simon HERMAN
Github: https://github.com/Sherman-1
======================================================
"""

import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import polars as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import compute_class_weight
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import evaluate

import wandb

# ========== Local imports ==========
import sys, os
from pathlib import Path
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

class CLASSIF_REG(nn.Module):
    def __init__(self, input_embed_dim: int, hidden_dim: int, num_classes: int, descriptors_dim: int, device, class_weights = None):
        super(CLASSIF_REG, self).__init__()
        self.device = device
        self.classifier = MLP(input_dim=input_embed_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
        self.regressor = MLP(input_dim=input_embed_dim, hidden_dim=hidden_dim, output_dim=descriptors_dim).to(device)
        
        if class_weights is None:
            self.class_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.classif_loss_fn = nn.BCEWithLogitsLoss(
                weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            )
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, embeddings, labels = None, phychem_descriptors = None):

        if embeddings is not None and embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        if phychem_descriptors is not None and phychem_descriptors.device != self.device:
            phychem_descriptors = phychem_descriptors.to(self.device)
        if labels is not None and labels.device != self.device:
            labels = labels.to(self.device)

        class_output = self.classifier(embeddings)  
        reg_output = self.regressor(embeddings)

        if labels is not None and phychem_descriptors is not None:

            loss_class = self.classif_loss_fn(class_output, labels)
            loss_reg = self.reg_loss_fn(reg_output, phychem_descriptors)
            combined_loss = loss_class + loss_reg

            return {
                "loss": combined_loss,
                "logits": class_output
            }
        
        else:
            return {"logits": class_output}

#####################################
# DATA LOADING 
#####################################

def get_input_data():
    print("Collecting training data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["descriptors", "protein_emb", "one_hot"])
        .collect()
        .rename({
            "protein_emb": "embeddings", 
            "one_hot": "labels", 
            "descriptors": "phychem_descriptors"
        })
        .to_pandas()
    )

    y_labels = np.argmax(np.stack(df["labels"].values), axis=1)
    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(y_labels),  
        y=y_labels  
    )
    print("    Building dataset ... ")
    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    # Keep labels as one-hot vectors.
    training_dataset.set_format("torch", columns=["phychem_descriptors", "embeddings", "labels"])

    print("Collecting eval data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["descriptors", "protein_emb", "one_hot"])
        .collect()
        .rename({
            "protein_emb": "embeddings", 
            "one_hot": "labels", 
            "descriptors": "phychem_descriptors"
        })
        .to_pandas()
    )
    print("    Building dataset ... ")
    eval_dataset = Dataset.from_pandas(df, preserve_index=False)
    eval_dataset.set_format("torch", columns=["phychem_descriptors", "embeddings", "labels"])

    return training_dataset, eval_dataset, loss_weights

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
# MAIN
#####################################

def main():

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory()
    print("     VRAM cleared!")

    # ========== Load / prepare data ==========
    
    train_ds, eval_ds, loss_weights = get_input_data()

    MODEL = CLASSIF_REG(
        input_embed_dim=1024, 
        hidden_dim=512, 
        num_classes=5, 
        descriptors_dim=83, 
        class_weights=loss_weights, 
        device=DEVICE
    ) 
    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/DB",
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
        run_name="DB_F1",
        logging_strategy="no"
        
    )

    trainer = Trainer(

        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,        
        compute_metrics=compute_metrics
    )

    wandb.init(project="DB", name="DB_F1_DeepSpeed")
    trainer.train()

if __name__ == "__main__":
    main()
