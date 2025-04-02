################

# 21/03/25 Use FT_MLP training on new dataset 

################

# ========== Local imports ==========
print("Importing modules")
from pathlib import Path
import sys, os

current_dir = Path(__file__).resolve()
for parent in current_dir.parents:
    if parent.name == "MicroPred":
        repo_root = parent
        break
else:
    raise RuntimeError("Repository root 'MicroPred' not found.")

src_path = repo_root / "src"
exp_path = repo_root / "experiment"
sys.path.append(str(src_path))
sys.path.append(str(exp_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"


from FT_MLP.FT_MLP import ProtT5Classifier as FT_MLP, LORA_CONFIG as FT_MLP_LORA

from datasets import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

import polars as pl 
from tqdm import tqdm 
from sklearn.metrics import f1_score, cohen_kappa_score

from collections import Counter

from peft import get_peft_model

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

from transformers import (
    T5Tokenizer,
    T5EncoderModel
)

import pandas as pd
from Bio import SeqIO

print("Modules imported")
print(f"{torch.__version__=}")
print(f"{pl.__version__=}")
print(f"{torch.cuda.is_available()=}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


def tokenize(examples):

    tokens = TOKENIZER(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=100
    )

    tokens["id"] = examples["id"]
    return tokens


def create_ft_model(model_class, lora_config, **kwargs):
    base_model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", 
        torch_dtype=torch.float32
    ).to(DEVICE)
    peft_model = get_peft_model(base_model, lora_config)
    return model_class(peft_model, **kwargs)

def get_models(): 

    return {

        "FT_MLP": {
            "type": "sequence",
            "model": create_ft_model(FT_MLP, FT_MLP_LORA, num_classes=5)
        }
    }


def get_input_data(fasta_path):

    """
    Load data for inference only
    Do not use this function for training, it does not split the data
    """

    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        spaced_sequence = " ".join(str(record.seq))
        sequences.append(spaced_sequence)
        ids.append(record.id)

    df = pl.DataFrame({
        "sequence": sequences, 
        "id" : ids
    })

    seq_df = df.select(["id","sequence"]).to_pandas()
    seq_dataset = Dataset.from_pandas(seq_df, preserve_index=False)
    seq_dataset = seq_dataset.map(tokenize, batched=True)
    seq_dataset = seq_dataset.remove_columns(["sequence"])
    seq_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask"], output_all_columns=True)

    return seq_dataset


def main(): 

    fastas = {

        "microproteins" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/Uniprot/Microproteins_big_sample.faa"
    }

    model_weights_path = "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/shufGlobMolt_FT_MLP/pytorch_model.bin"
    state_dict = torch.load(model_weights_path)
    state_dict.pop("loss_fn.weight", None)
    state_dict.pop("classif_loss_fn.weight", None)

    model = get_models()["FT_MLP"]["model"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    dfs = []

    for GC_percent, fasta_path in fastas.items():

        print(f"Loading {fasta_path}")

        dataset = get_input_data(fasta_path)

        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False) 
        
        print("Dataloading done ! ")

        logits_list = []
        ids = []
        print("Starting inference")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc = "Ola tornado", total = len(dataloader)):
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "id"}
                outputs = model.forward(**inputs)
                logits = outputs["logits"]
                ids.extend(batch["id"])
                logits_list.extend(logits.cpu().tolist())

        dfs.append(pl.DataFrame({
            "id": ids,
            "logits": logits_list,
            "category": [GC_percent] * len(ids)
        }))

    df = pl.concat(dfs)
    df.write_parquet("big_shufGlobMolt_microproteins_predictions.parquet")


if __name__ == "__main__":

    main()
    