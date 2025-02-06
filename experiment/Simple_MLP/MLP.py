import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MLP, INT_TO_CLASS
from utils import Trainer
from data_utils import get_dataloaders

import polars as pl 
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

import copy


def main():

    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

    train_dl, val_dl, class_weights = get_dataloaders()

    model = MLP(1024, len(class_weights)).to(device)

    params = {

        "train_dl" : train_dl,
        "val_dl" : val_dl,
        "model" : model,
        "loss_fn" : nn.CrossEntropyLoss(class_weights).to(device),
        "optimizer" : torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
        "epochs" : 100,
        "scheduler" : None,
        "logging" : True,
        "val_interval_batches" : 10,
        "grad_accum_steps" : 1, # Just don't
        "output_path" : None

    }

    wandb_project = "Simple MLP"
    
    trainer = Trainer(**params, wandb_config=params, wandb_project=wandb_project)

    trainer.train()

if __name__ == "__main__":

    main()