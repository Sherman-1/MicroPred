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

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from peft import get_peft_model, LoraConfig, TaskType

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

LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)


#####################################
# MODEL DEFINITION
#####################################

class FT_CLASSIF_REG_MHAP(nn.Module):
    """
    ProtT5 encoder + MultiHeadAttention pooling of embeddings instead of mean pooling + MLP classifier/regressor
    """
    def __init__(self, base_model, input_embed_dim: int, output_embed_dim: int, hidden_dim: int, num_classes: int, descriptors_dim: int, class_weights, device):
        super(FT_CLASSIF_REG_MHAP, self).__init__()
        self.device = device
        self.encoder = base_model.to(device) 
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim).to(device)
        self.classifier = MLP(input_dim=output_embed_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
        self.regressor = MLP(input_dim=output_embed_dim, hidden_dim=hidden_dim, output_dim=descriptors_dim).to(device)
        self.classif_loss_fn = nn.BCEWithLogitsLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        )
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels, phychem_descriptors):

        residue_embeddings = self.encoder(input_ids, attention_mask).last_hidden_state # (batch_size, seq_len, embed_dim)
        custom_mhap_masks = ~attention_mask.bool()  # (batch_size, seq_len)
        pooled_embeddings, _ = self.mhap(residue_embeddings, custom_mhap_masks) # (batch_size, embed_dim)
        class_output = self.classifier(pooled_embeddings) # (batch_size, num_classes)
        reg_output = self.regressor(pooled_embeddings) # (batch_size, descriptors_dim)


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

TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

def tokenize(examples):

    tokens = TOKENIZER(
        examples["sequence"],
        padding="longest", # Do it on the fly
        truncation=True,
        max_length=1024
    )
    tokens["labels"] = examples["labels"]
    return tokens

def get_input_data():

    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["sequence", "category", "one_hot", "descriptors"])
        .rename({
            "one_hot": "labels",
            "descriptors": "phychem_descriptors"
        })
    )

    y = df.select("category").collect().to_series().to_numpy()
    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(y), 
        y=y
    )
    training_dataset = Dataset.from_pandas(df.collect().to_pandas(), preserve_index=False)
    training_dataset = training_dataset.map(tokenize, batched=True)
    training_dataset = training_dataset.remove_columns(["sequence", "category"])
    training_dataset.set_format("torch")

    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["sequence", "category","one_hot", "descriptors"])
        .rename({
            "one_hot": "labels",
            "descriptors": "phychem_descriptors"
        })
    )

    eval_dataset = Dataset.from_pandas(df.collect().to_pandas(), preserve_index=False)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.remove_columns(["sequence", "category"])
    eval_dataset.set_format("torch")

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

    train_dl = DataLoader(train_ds, batch_size=10)

    # ========== Load / prepare model ==========

    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    MODEL = FT_CLASSIF_REG_MHAP(

        base_model=PEFT_MODEL,
        input_embed_dim=1024,
        output_embed_dim=512,
        hidden_dim=256,
        num_classes=5,
        descriptors_dim=83,
        class_weights=loss_weights,
        device=DEVICE

    )

    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/FT_DB_MHAP",
        num_train_epochs=3,
        
        per_device_train_batch_size=128,
        per_device_eval_batch_size=512,
        eval_strategy="steps",
        remove_unused_columns=False,  
        
        fp16=True,
        deepspeed="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ds_config.json",
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",   
        greater_is_better=True,
        save_strategy="steps",
        save_total_limit=1,
        logging_steps=100,
        report_to=["wandb"],
        run_name="FT_DB_MHAP",
        logging_strategy = "steps"

    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds, 
        compute_metrics=compute_metrics  
    )

    wandb.init(project="FT_DB_MHAP", name="FT_DB_MHAP_DeepSpeed")

    trainer.train()

if __name__ == "__main__":
    main()
