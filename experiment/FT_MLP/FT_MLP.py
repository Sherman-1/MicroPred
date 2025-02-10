
import random
import numpy as np 
import polars as pl 

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils import compute_class_weight

from transformers import T5Tokenizer, T5EncoderModel, set_seed
from peft import  get_peft_model, LoraConfig, TaskType
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MLP
from utils import print_gpu_memory, check_model_on_gpu, Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not torch.cuda.is_available() : exit("Cuda compatible GPU not found")


class ProtT5Classifier(nn.Module):

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.encoder = base_model
        self.classifier = MLP(input_dim = 1024, hidden_dim = 512, output_dim=num_classes)  

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state.mean(dim=1))
        return logits


def main():

    torch.cuda.empty_cache()  
    torch.cuda.synchronize()  
    print_gpu_memory()
    print("     VRAM cleared!")

    LORA_CONFIG = LoraConfig(
        r=4,  
        lora_alpha=32,  
        lora_dropout=0.1,  
        target_modules=["q", "v"],  
        bias="none",  
        task_type=TaskType.FEATURE_EXTRACTION  
    )

    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    MODEL = ProtT5Classifier(PEFT_MODEL, 5).to(DEVICE)
    TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    check_model_on_gpu(MODEL)

    print(f"Loading data ...")

    ######### TRAIN DATA #########
    print("     Training set ...")
    train_data = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train_sequences.csv")
    tokenized = TOKENIZER(
        train_data["sequence"].to_list(), 
        max_length=1024, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    dataset = TensorDataset(
        tokenized["input_ids"], 
        train_data["category"].to_torch().long(),
        F.one_hot(train_data["category"].to_torch(), num_classes=5).float(),
        tokenized["attention_mask"],
        torch.zeros(train_data["category"].to_torch().shape[0]) # Fake sequence ID just for the DataLoader
    )

    train_dl = DataLoader(dataset, batch_size=8, shuffle=True)
    try:
        next(iter(train_dl))
    except Exception as e:
        print("Error while building training dataloader :")
        print(e)

    cw = compute_class_weight(y = train_data["category"].to_list(), class_weight = "balanced", classes = np.sort((train_data["category"].unique().to_list()))) # Sort to match CLASS_TO_INT)
    cw = torch.tensor(cw, dtype=torch.float32)

    ######### EVAL DATA #########
    print("     Validation set ...")
    val_data = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/test_sequences.csv")
    tokenized = TOKENIZER(
        val_data["sequence"].to_list(), 
        max_length=1024, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )

    dataset = TensorDataset(
        tokenized["input_ids"], 
        val_data["category"].to_torch().long(),
        F.one_hot(val_data["category"].to_torch(), num_classes=5).float(),
        tokenized["attention_mask"],
        torch.zeros(val_data["category"].to_torch().shape[0])

    )
    val_dl = DataLoader(dataset, batch_size=126, shuffle=True)

    try:
        next(iter(val_dl))
    except Exception as e:
        print("Error while building validation dataloader :")
        print(e)

    optimizer = torch.optim.AdamW(MODEL.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    params = {

        "train_dl" : train_dl,
        "val_dl" : val_dl,
        "model" : MODEL,
        "loss_fn" : nn.CrossEntropyLoss(weight=torch.as_tensor(cw, device=DEVICE, dtype=torch.float32)),
        "optimizer" : optimizer,
        "epochs" : 50,
        "scheduler" : scheduler,
        "logging" : True,
        "grad_accum_steps" : 10,
        "output_path" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/FT_MLP.pth",
        "debug" : True,
        "use_amp" : True

    }

    wandb_project = "FT-MLP"
        
    print(f"Starting training ... ")

    trainer = Trainer(**params, wandb_config=params, wandb_project=wandb_project)

    trainer.train()


if __name__ == "__main__":

    import argparse
    import doctest
    import sys

    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)   

                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )

        sys.exit()

    main()

