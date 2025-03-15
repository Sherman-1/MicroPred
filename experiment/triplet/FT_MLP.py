#!/usr/bin/env python3
"""
======================================================
Author: Simon HERMAN
Github: https://github.com/Sherman-1
======================================================
"""

import warnings
import logging

# Disable warnings and logging from HuggingFace
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
from torch.nn.utils.rnn import pad_sequence
import torch

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
)

import wandb

# ========== Local imports ==========
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
src_path = root_dir / 'src'
sys.path.append(str(src_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

from MHA import MLP  
from utils import print_gpu_memory, check_model_on_gpu, set_seed


from peft import get_peft_model, LoraConfig, TaskType

#####################################
# GLOBALS / CONFIG
#####################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LORA_CONFIG = LoraConfig(
    r=2,
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
# DATA LOADING & TOKENIZATION
#####################################

def tokenize(examples):
    tokens = TOKENIZER(
        examples["sequence"],
        padding="max_length",  # use max_length to ensure fixed-length inputs
        truncation=True,
        max_length=100
    )
    tokens["labels"] = examples["labels"]
    return tokens

def get_input_data():
    # Load training data from parquet and process it into a Hugging Face Dataset.
    df = (
        pl.scan_parquet("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet")
        .select(["sequence", "category", "one_hot"])
        .rename({"one_hot": "labels"})
        .collect()
        .to_pandas()
    )
    from datasets import Dataset  # import here to avoid circular issues
    training_dataset = Dataset.from_pandas(df, preserve_index=False)
    training_dataset = training_dataset.map(tokenize, batched=True)
    training_dataset = training_dataset.remove_columns(["sequence", "category"])
    training_dataset.set_format("torch")
    
    return training_dataset

#####################################
# CUSTOM TRIPLET DATASET & COLLATOR
#####################################

class TripletHFDataset(torch.utils.data.Dataset):
    """
    Wraps a Hugging Face dataset to sample triplets.
    For each anchor, a positive (same label) and negative (different label) are sampled.
    """
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.label_to_indices = {}
        for i, example in enumerate(self.dataset):
            label = int(torch.argmax(example["labels"])) if isinstance(example["labels"], torch.Tensor) else int(np.argmax(example["labels"]))
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(i)

            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor = self.dataset[idx]
        label = int(torch.argmax(anchor["labels"])) if isinstance(anchor["labels"], torch.Tensor) else int(np.argmax(anchor["labels"]))
        
        # Sample a positive example (same label but a different instance)
        positive_idx = random.choice(self.label_to_indices[label])
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[label])
        positive = self.dataset[positive_idx]
        
        # Sample a negative example (different label)
        negative_label = random.choice([l for l in self.label_to_indices.keys() if l != label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative = self.dataset[negative_idx]
        
        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }


class TripletDataCollator:
    """
    Custom collator to pad anchor, positive, and negative fields.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        anchors_input_ids = []
        anchors_attention_mask = []
        positives_input_ids = []
        positives_attention_mask = []
        negatives_input_ids = []
        negatives_attention_mask = []

        for f in features:
            anchors_input_ids.append(torch.tensor(f["anchor_input_ids"], dtype=torch.long))
            anchors_attention_mask.append(torch.tensor(f["anchor_attention_mask"], dtype=torch.long))
            positives_input_ids.append(torch.tensor(f["positive_input_ids"], dtype=torch.long))
            positives_attention_mask.append(torch.tensor(f["positive_attention_mask"], dtype=torch.long))
            negatives_input_ids.append(torch.tensor(f["negative_input_ids"], dtype=torch.long))
            negatives_attention_mask.append(torch.tensor(f["negative_attention_mask"], dtype=torch.long))

        batch = {
            "anchor_input_ids": pad_sequence(anchors_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "anchor_attention_mask": pad_sequence(anchors_attention_mask, batch_first=True, padding_value=0),
            "positive_input_ids": pad_sequence(positives_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "positive_attention_mask": pad_sequence(positives_attention_mask, batch_first=True, padding_value=0),
            "negative_input_ids": pad_sequence(negatives_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "negative_attention_mask": pad_sequence(negatives_attention_mask, batch_first=True, padding_value=0),
        }
        return batch


#####################################
# TRIPLET MODEL DEFINITION
#####################################
class ProtT5TripletModel(nn.Module):
    """
    pT5 encoder + projection head (MLP) for triplet loss training.
    Computes embeddings for anchor, positive, and negative inputs.
    """
    def __init__(self, base_model, margin=1.0):
        super().__init__()
        self.encoder = base_model
        # Projection head: adjust output_dim as needed
        self.projection = MLP(input_dim=1024, hidden_dim=512, output_dim=128)
        self.margin = margin
    
    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        **kwargs
    ):
        anchor_outputs = self.encoder(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask
        )
        anchor_embed = anchor_outputs.last_hidden_state.mean(dim=1)
        anchor_proj = self.projection(anchor_embed)
        
        positive_outputs = self.encoder(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask
        )
        positive_embed = positive_outputs.last_hidden_state.mean(dim=1)
        positive_proj = self.projection(positive_embed)
        
        negative_outputs = self.encoder(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask
        )
        negative_embed = negative_outputs.last_hidden_state.mean(dim=1)
        negative_proj = self.projection(negative_embed)
        
        pos_dist = torch.norm(anchor_proj - positive_proj, p=2, dim=1)
        neg_dist = torch.norm(anchor_proj - negative_proj, p=2, dim=1)
        
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        
        # Return "loss" so HF Trainer can do backprop
        return {"loss": loss}


#####################################
# MAIN TRAINING PIPELINE FOR TRIPLET LOSS
#####################################

def main():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory()
    print("VRAM cleared!")

    # ========== Load / prepare data ==========
    train_ds = get_input_data()
    triplet_train_ds = TripletHFDataset(train_ds)
    data_collator = TripletDataCollator(TOKENIZER)



    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", 
            torch_dtype=torch.float16
        ).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    
    model = ProtT5TripletModel(PEFT_MODEL, margin=1.0).to(DEVICE)
    check_model_on_gpu(model)

    training_args = TrainingArguments(
        output_dir="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/Triplet",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_strategy="steps",
        save_total_limit=1,
        report_to=["wandb"],
        run_name="Triplet_Training",

        fp16=True,
        deepspeed="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/ds_config.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=triplet_train_ds,
        data_collator=data_collator,
        # No compute_metrics: here we optimize the triplet loss.
    )

    wandb.init(project="Triplet_Training", name="Triplet_DeepSpeed")
    trainer.train()

if __name__ == "__main__":
    main()
