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

from MHA import MLP, MHAPooling
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

class CLASSIF_REG_MHAP(nn.Module):
    def __init__(self, input_embed_dim: int, output_embed_dim: int, hidden_dim: int, num_classes: int, descriptors_dim: int, class_weights, device):
        super(CLASSIF_REG_MHAP, self).__init__()
        self.device = device
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim).to(device)
        self.classifier = MLP(input_dim=output_embed_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
        self.regressor = MLP(input_dim=output_embed_dim, hidden_dim=hidden_dim, output_dim=descriptors_dim).to(device)
        self.classif_loss_fn = nn.BCEWithLogitsLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        )
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, embeddings, attention_mask, labels, phychem_descriptors):

        if embeddings is not None and embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        if phychem_descriptors is not None and phychem_descriptors.device != self.device:
            phychem_descriptors = phychem_descriptors.to(self.device)
        if labels is not None and labels.device != self.device:
            labels = labels.to(self.device)
        if attention_mask is not None and attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        attn_output, _ = self.mhap(embeddings, attention_mask)  
        class_output = self.classifier(attn_output) 
        reg_output = self.regressor(attn_output) 

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
        .select(["descriptors", "residue_emb", "one_hot"])
        .rename({
            "residue_emb": "embeddings", 
            "one_hot": "labels", 
            "descriptors": "phychem_descriptors"
        })
    )

    y_labels = np.argmax(np.stack(df.select("labels").collect().to_series().to_numpy()), axis=1)
    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(y_labels),
        y=y_labels  
    )
    print("    Building dataset ... ")
    training_dataset = Dataset.from_pandas(df.collect().to_pandas(), preserve_index=False)
    training_dataset.set_format("torch", columns=["phychem_descriptors", "embeddings", "labels"])

    print("Collecting eval data")
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["descriptors", "residue_emb", "one_hot"])
        .rename({
            "residue_emb": "embeddings", 
            "one_hot": "labels", 
            "descriptors": "phychem_descriptors"
        })
    )
    print("    Building dataset ... ")
    eval_dataset = Dataset.from_pandas(df.collect().to_pandas(), preserve_index=False)
    eval_dataset.set_format("torch", columns=["phychem_descriptors", "embeddings", "labels"])

    return training_dataset, eval_dataset, loss_weights


def padding_collate(batch):

    embeddings = [item["embeddings"] for item in batch]
    labels = [torch.as_tensor(item["labels"]) for item in batch]  
    phychem_descriptors = [torch.as_tensor(item["phychem_descriptors"]) for item in batch]
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
    batch_phychem_descriptors = torch.stack(phychem_descriptors)
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
        "attention_mask": batch_attention_mask,
        "phychem_descriptors": batch_phychem_descriptors
    }

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

    MODEL = CLASSIF_REG_MHAP(
        input_embed_dim = 1024, 
        output_embed_dim = 512,
        hidden_dim = 256,
        num_classes = 5,
        descriptors_dim = 83,
        class_weights = loss_weights,
        device = DEVICE
    )

    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/DB_MHAP",
        num_train_epochs=100,
        
        per_device_train_batch_size=512,
        per_device_eval_batch_size=1024,
        eval_strategy="epoch",
        remove_unused_columns=False,  
        
        fp16=True,
        deepspeed="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ds_config.json",
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",   
        greater_is_better=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to=["wandb"],
        run_name="DB_MHAP",
        logging_strategy="no"
        
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,      
        data_collator=padding_collate,
        compute_metrics=compute_metrics  
    )

    wandb.init(project="DB_MHAP", name="DB_MHAP_F1")

    trainer.train()

if __name__ == "__main__":
    main()
