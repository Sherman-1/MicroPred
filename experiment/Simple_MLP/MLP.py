import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MLP, INT_TO_CLASS
from utils import Trainer
from data_utils import EmbedProtT5Dataset

import polars as pl 
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

train_ds = EmbedProtT5Dataset(ptfile="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/testset_protein_embeddings.pt",
                        num_classes = 5)

val_ds = EmbedProtT5Dataset(ptfile="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/testset_protein_embeddings.pt",
                        num_classes = 5)

class_weight_dict = val_ds.class_weights 




train_dl = DataLoader(train_ds, batch_size = 20000, shuffle = True)
val_dl = DataLoader(val_ds, batch_size=1024)


input_shape = next(iter(train_dl))[0].shape[1] 
output_shape = len(class_weight_dict)


model = MLP(input_shape, output_shape).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

for batch in train_dl: 

    embedding, one_hot, class_type, name = batch 

    # Assume embeddings is a (batch_size, embedding_dim) tensor
    embedding_norms = torch.norm(embedding, p=2, dim=1)  # Compute L2 norm for each embedding
    print("Mean norm:", embedding_norms.mean().item())
    print("Std dev of norm:", embedding_norms.std().item())

    break



"""
 self,
        model,
        train_dl,
        val_dl=None,
        optimizer=None,
        loss_fn=None,
        epochs=10,
        scheduler=None,
        device=None,
        logging=False,
        wandb_project="default_project",
        wandb_config=None,
        val_interval_batches=None,  # Compute validation every N batches if set
        grad_accum_steps=1,         # Number of mini-batches to accumulate gradients over
        output_path = None,
"""
