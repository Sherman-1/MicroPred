#!/usr/bin/env python3


"""
======================================================
Author: Simon HERMAN
Github: https://github.com/Sherman-1
======================================================
"""

import warnings
import logging

# F em huggingface logs
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


import os
import sys
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

import evaluate

from peft import get_peft_model, LoraConfig, TaskType
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

# ===================================

#####################################
# GLOBALS / CONFIG
#####################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_CONFIG = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

if not torch.cuda.is_available():
    sys.exit("Cuda-compatible GPU not found!")

set_seed(66)


TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


#####################################
# MODEL DEFINITION
#####################################

class ProtT5Classifier(nn.Module):
    """
    pT5 encoder + One Hidden Layer Perceptron
    Simple mean pool on residue embeddings
    """
    def __init__(self, base_model, num_classes, loss_weights=None):
        super().__init__()
        self.encoder = base_model
        self.classifier = MLP(input_dim=1024, hidden_dim=512, output_dim=num_classes)

        if loss_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                weight=torch.as_tensor(loss_weights, device=base_model.device, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Average pooling over seq length ( L )
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits


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
        .select(["sequence", "category", "one_hot"])
        .rename({"one_hot": "labels"})
        .collect()
        .to_pandas()
    )

    loss_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(df["category"].values), 
        y=df["category"].values
    )

    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    training_dataset = training_dataset.map(tokenize, batched=True)
    training_dataset = training_dataset.remove_columns(["sequence", "category"])
    training_dataset.set_format("torch")

    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/eval.parquet")
        .select(["sequence", "category","one_hot"])
        .rename({"one_hot": "labels"})
        .collect()
        .to_pandas()
    )

    eval_dataset = Dataset.from_pandas(df, preserve_index=False)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.remove_columns(["sequence", "category"])
    eval_dataset.set_format("torch")

    return training_dataset, eval_dataset, loss_weights


#####################################
# MAIN TRAINING PIPELINE
#####################################

def main():

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory()
    print("     VRAM cleared!")

    # ========== Load / prepare data ==========
    
    train_ds, eval_ds, loss_weights = get_input_data()

    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    MODEL = ProtT5Classifier(PEFT_MODEL, 5, loss_weights = loss_weights).to(DEVICE)
    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/FT_MLP",
        num_train_epochs=5,
        
        per_device_train_batch_size=16,
        per_device_eval_batch_size=126,
        evaluation_strategy="steps",
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
        run_name="FT_DeepSpeed"
    )


    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,      
        data_collator=data_collator,     
        compute_metrics=compute_metrics  
    )

    wandb.init(project="FT_MLP", name="FT_DeepSpeed")

    trainer.train()

if __name__ == "__main__": 

    main()