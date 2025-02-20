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

from MHA import MLP
from utils import print_gpu_memory, check_model_on_gpu, set_seed
from Simple_MLP.MLP import MLP as MLP 

import torch.nn as nn
import torch

from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

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


def get_predictions( 
    model: nn.Module,
    input_data: torch.Tensor,
    device,
    model_weights_path,
    mode_type: str
                    ): 

    """
    Model is either expecting : 
        - tokenized sequences ( input_ids + attention_mask )
        - mean pooled embeddings ( 1x1024 tensor )
        - full residue embeddings ( Lx1024 tensor + attention_mask )
    """

    assert mode_type in ["tokenized", "mean_pooled", "full_residue"], "Invalid mode type"

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)    

    if model_type == "tokenized":
        input_ids, attention_mask = input_data
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
    




