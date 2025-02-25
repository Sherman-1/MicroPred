# ========== Local imports ==========
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

from utils import check_model_on_gpu

from MLP.MLP import ProtT5Classifier as MLP
from DB.DB import CLASSIF_REG as DB
from DB_MHAP.DB_MHAP import CLASSIF_REG_MHAP as DB_MHAP
from MHAP_MLP.MHAP_MLP import MHAP_MLP

from FT_MLP.FT_MLP import ProtT5Classifier as FT_MLP, LORA_CONFIG as FT_MLP_LORA
from FT_DB.FT_DB import REG_CLASS as FT_DB, LORA_CONFIG as FT_DB_LORA
from FT_DB_MHAP.FT_DB_MHAP import FT_CLASSIF_REG_MHAP as FT_DB_MHAP, LORA_CONFIG as FT_DB_MHAP_LORA
from FT_MHAP.FT_MHAP import FT_MHAP_MLP, LORA_CONFIG as FT_MHAP_MLP_LORA

from MHA import INT_TO_CLASS, CLASSES, CLASS_TO_INT

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

MODELS_WEIGHTS_BASE_PATH = Path("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models")

def get_models(): 

    return {

        "MLP": {
            "type": "protein_emb",
            "model": MLP(num_classes=5).to(DEVICE)
        },
        "DB": {
            "type": "protein_emb",
            "model": DB(input_embed_dim=1024, hidden_dim=512, num_classes=5,
                            descriptors_dim=83, device=DEVICE).to(DEVICE)
        },


        "DB_MHAP": {
            "type": "residue_emb",
            "model": DB_MHAP(input_embed_dim=1024, output_embed_dim=512, hidden_dim=256,
                            num_classes=5, descriptors_dim=83, device=DEVICE).to(DEVICE)
        },

        "MHAP_MLP": {
            "type": "residue_emb",
            "model": MHAP_MLP(input_embed_dim=1024, output_embed_dim=256,
                            mlp_hidden_dim=128, num_classes=5).to(DEVICE)
        },

        "FT_MLP": {
            "type": "sequence",
            "model": create_ft_model(FT_MLP, FT_MLP_LORA, num_classes=5)
        },
        "FT_DB": {
            "type": "sequence",
            "model": create_ft_model(FT_DB, FT_DB_LORA, num_classes=5, descriptors_dim=83, device = DEVICE)
        },
        "FT_DB_MHAP": {
            "type": "sequence",
            "model": create_ft_model(FT_DB_MHAP, FT_DB_MHAP_LORA, num_classes=5,
                                    input_embed_dim=1024,
                                    output_embed_dim=512,
                                    hidden_dim=256,
                                    descriptors_dim=83,
                                    device=DEVICE
                                )
        },
        "FT_MHAP_MLP": {
            "type": "sequence",
            "model": create_ft_model(FT_MHAP_MLP, FT_MHAP_MLP_LORA, num_classes=5,
                                    input_embed_dim=1024,
                                    output_embed_dim=512,
                                    hidden_dim=256,
                                    device=DEVICE
                                )
        }
        
    }

def tokenize(examples):

    tokens = TOKENIZER(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=101 # Microproteins baby
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

def old_padding_collate(batch):

    embeddings = [item["embeddings"] for item in batch]
    ids = [item["id"] for item in batch]
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
    batch_attention_mask = torch.stack(attention_masks)
    batch_attention_mask = ~batch_attention_mask  

    return {
        "embeddings": batch_embeddings,
        "attention_mask": batch_attention_mask,
        "id": ids
    }
def padding_collate(batch):

    embeddings = [
        torch.tensor(item["embeddings"]) if not isinstance(item["embeddings"], torch.Tensor)
        else item["embeddings"]
        for item in batch
    ]
    
    max_len = max(emb.shape[0] for emb in embeddings)
    print("Max length in current batch:", max_len)
    
    padded_embeddings = []
    attention_masks = []
    
    for emb in embeddings:
        seq_len, emb_dim = emb.shape
        pad_len = max_len - seq_len
        if pad_len > 0:
            padded_emb = F.pad(emb, (0, 0, 0, pad_len))
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.bool), 
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            padded_emb = emb
            mask = torch.ones(seq_len, dtype=torch.bool)
        
        padded_embeddings.append(padded_emb)
        attention_masks.append(mask)
    
    # Debug: confirm all padded embeddings have the same shape
    for i, padded in enumerate(padded_embeddings):
         if padded.shape[0] != max_len:
            print(f"Error: Padded embedding {i} has shape {padded.shape}")

            for j,padded in enumerate(padded_embeddings):
                print(f"Embedding {j} shape: {padded.shape}")
            raise ValueError("Padded embeddings have different shapes")
    
    batch_embeddings = torch.stack(padded_embeddings)
    batch_attention_mask = torch.stack(attention_masks)
    batch_attention_mask = ~batch_attention_mask
    
    return {
        "embeddings": batch_embeddings,
        "attention_mask": batch_attention_mask,
        "id": [item["id"] for item in batch]
    }


def prepare_datasets(data_path):

    df = (
        pl.scan_parquet(data_path)
        .select(["id","sequence", "protein_emb", "residue_emb"])
        .collect()
    )
    
    datasets = {}

    seq_df = df.select(["id","sequence"]).to_pandas()
    seq_dataset = Dataset.from_pandas(seq_df, preserve_index=False)
    seq_dataset = seq_dataset.map(tokenize, batched=True)
    seq_dataset = seq_dataset.remove_columns(["sequence"])
    seq_dataset.set_format(type = "torch", columns = ["input_ids", "attention_mask"], output_all_columns=True)
    datasets["sequence"] = seq_dataset

    prot_df = df.select(["id","protein_emb"]).rename({"protein_emb": "embeddings"}).to_pandas()
    prot_dataset = Dataset.from_pandas(prot_df, preserve_index=False)
    prot_dataset.set_format("torch", columns=["embeddings"], output_all_columns=True)
    datasets["protein_emb"] = prot_dataset

    resid_df = df.select(["id", "residue_emb"]).rename({"residue_emb": "embeddings"}).to_pandas()
    resid_dataset = Dataset.from_pandas(resid_df, preserve_index=False)
    resid_dataset.set_format("torch", columns=["embeddings"], output_all_columns=True)
    datasets["residue_emb"] = resid_dataset

    return datasets

real_data_path = {
    "Swissprot_microproteins": "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/predict_data/Swiss_microproteins.parquet",
    "Trembl_microproteins": "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/predict_data/Trembl_microproteins.parquet",
    "Scer_iORFs": "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/predict_data/Scer_iORFs.parquet"
}

LOGITS_DF_FILE = "logits_dataframe.parquet"

def compute_logits_dataframes(models):

    dataframes = list()
    
    for data_type, data_path in real_data_path.items():
        print(f"Processing data type: {data_type}")
        datasets = prepare_datasets(data_path)
        all_logits = {}
        common_ids = None  
        
        for model_name, model_infos in models.items():

            print(f"   Processing model: {model_name}")
            model_type = model_infos["type"]
            print(f"    Model type : {model_type}")
            dataset = datasets[model_type]

            print(dataset)
            
            if model_type == "residue_emb":
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=padding_collate)
            elif model_type == "sequence":
                dataloader = DataLoader(dataset, batch_size=512, shuffle=False) 
            else: 
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
            
            ids = []

            print(f"   Computing logits for model: {model_name} on {data_type}")
            model_weights_path = MODELS_WEIGHTS_BASE_PATH / model_name / "pytorch_model.bin"
            state_dict = torch.load(model_weights_path)
            state_dict.pop("loss_fn.weight", None)
            state_dict.pop("classif_loss_fn.weight", None)
            
            model = model_infos["model"]
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            
            logits_list = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "id"}
                    outputs = model.forward(**inputs)
                    ids.extend(batch["id"])
                    logits = outputs["logits"]
                    logits_list.extend(logits.cpu().tolist())
            
            all_logits[model_name] = logits_list
            if common_ids is None:
                common_ids = ids
        
        df_dict = {"id": common_ids}
        df_dict.update(all_logits)
        dataframes.append(pl.DataFrame(df_dict).with_columns(database = pl.lit(data_type)))

    return pl.concat(dataframes)

if __name__ == "__main__":

    if not os.path.exists(LOGITS_DF_FILE):

        models = get_models()
        print("Computing logits dataframes...")
        logits_dataframes = compute_logits_dataframes(models)
        logits_dataframes.write_parquet(LOGITS_DF_FILE, compression = "lz4")

    else:

        print("Logits already computed, delete file and rerun script to recompute.")

