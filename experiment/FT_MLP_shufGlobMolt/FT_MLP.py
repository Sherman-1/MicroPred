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
from datasets import Dataset, load_from_disk
import wandb

# ========== Local imports ==========
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_path = root_dir / 'src'
sys.path.append(str(src_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

from MHA import MLP
from utils import print_gpu_memory, check_model_on_gpu, set_seed

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from sklearn.model_selection import train_test_split

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

set_seed(66370)


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
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) # ( B, L, 1024 )
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Average pooling over seq length ( L )
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            labels = labels.to(dtype=logits.dtype)
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

def tokenize(examples):

    tokens = TOKENIZER(
        examples["sequence"],
        padding="longest", # Do it on the fly
        truncation=True,
        max_length=100
    )
    tokens["labels"] = examples["labels"]
    return tokens


def get_input_data():

    fastas = {
        
        "globular" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas_GlobMoltenShuf/split_globular.part_001.fasta",
        "molten" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas_GlobMoltenShuf/split_molten.part_001.fasta",
        "transmembrane" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas_GlobMoltenShuf/transmembrane_elongated_representatives.fasta",
        "disordered" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas_GlobMoltenShuf/representative_disordered_sequences.fasta",
        "shuffled_GlobMolt" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/new_processed_fastas_GlobMoltenShuf/shuffled_GlobMolt.fasta"
    }

    CLASSES = [
        "molten",
        "globular",
        "transmembrane",
        "disordered",
        "shuffled_GlobMolt"
    ]

    CLASS_TO_INT = dict(zip(CLASSES, range(len(CLASSES)))) 

    sequences = []
    categories = []
    ids = []

    for category, fasta_file in fastas.items():
        for record in SeqIO.parse(fasta_file, "fasta"):
            spaced_sequence = " ".join(str(record.seq))
            sequences.append(spaced_sequence)
            categories.append(category)
            ids.append(record.id)

    df = pd.DataFrame({
        "seq_id": ids,
        "sequence": sequences,
        "category": categories
    })

    def one_hot_encode(cat):
        vector = [0] * len(CLASSES)
        vector[CLASS_TO_INT[cat]] = 1
        return vector

    def save_fasta(df, filepath):
        records = []
        for _, row in df.iterrows():
            seq = row["sequence"].replace(" ", "")  # remove spacing
            record = SeqRecord(Seq(seq), id=row["seq_id"], description=row["category"])
            records.append(record)
        SeqIO.write(records, filepath, "fasta")

    df["labels"] = df["category"].apply(one_hot_encode)

    train_df, eval_df = train_test_split(
        df,
        test_size=0.5,
        stratify=df["category"],
        random_state=42
    )

    save_fasta(train_df, "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/shufGlobMolt_train_set.fasta")
    save_fasta(eval_df, "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/shufGlobMolt_eval_set.fasta")

    loss_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["category"].values),
        y=df["category"].values
    )

    print("Loss weights: ", loss_weights)
    print("Training set size: ", len(train_df))
    print("Eval set size: ", len(eval_df))
    print("Training set distribution: ", train_df["category"].value_counts())
    print("Eval set distribution: ", eval_df["category"].value_counts())

    training_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    training_dataset = training_dataset.map(tokenize, batched=True)
    training_dataset = training_dataset.remove_columns(["sequence", "category","seq_id"])
    training_dataset.set_format(type = "torch")

    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.remove_columns(["sequence", "category","seq_id"])
    eval_dataset.set_format(type = "torch")


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

    # Later load them with datasets' load_from_disk
    train_ds.save_to_disk("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/shufGlobMolt_training_dataset")
    eval_ds.save_to_disk("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/shufGlobMolt_eval_dataset")

    # train_ds = load_from_disk("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/new_training_dataset")
    # eval_ds = load_from_disk("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/new_eval_dataset")

    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    MODEL = ProtT5Classifier(PEFT_MODEL, 5, loss_weights = loss_weights).to(DEVICE)
    check_model_on_gpu(MODEL)

    training_args = TrainingArguments(

        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/shufGlobMolt_FT_MLP",
        num_train_epochs=3,
        
        per_device_train_batch_size=32,
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
        logging_steps=500,
        save_steps=500,
        report_to=["wandb"],
        run_name="shufGlobMolt_FT_MLP"

    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=eval_ds,      
        data_collator=data_collator,     
        compute_metrics=compute_metrics  
    )

    wandb.init(project="shufGlobMolt_FT_MLP", name="FT_DeepSpeed")

    trainer.train()

if __name__ == "__main__": 

    main()