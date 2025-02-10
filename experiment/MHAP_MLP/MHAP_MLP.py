import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MHAPooling, MLP, INT_TO_CLASS
from utils import Trainer
from data_utils import get_dataloaders, padding_collate

import polars as pl 
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn

class MHAP_MLP(nn.Module):
    """
    forward batch:
    >>> batch = torch.rand([64, 36, 1024])
    >>> batch.shape
    torch.Size([64, 36, 1024])
    >>> mhap_mlp = MHAP_MLP(input_embed_dim = 1024, output_embed_dim = 128, num_classes = 5)
    >>> mlp_output = mhap_mlp.forward(batch, mask = None)
    >>> mlp_output.shape
    torch.Size([64, 5])
    """
    def __init__(self, input_embed_dim: int, num_classes: int, output_embed_dim:int=None):
        super(MHAP_MLP, self).__init__()
        self.mhap = MHAPooling(embed_dim=input_embed_dim, d_out = output_embed_dim)
        self.mlp = MLP(input_dim=output_embed_dim, output_dim=num_classes, hidden_dim = 256)

    def forward(self, embeddings, mask):
        attn_output, _ = self.mhap(embeddings, mask)  
        mlp_output = self.mlp(attn_output)             
        return mlp_output
    

def main():

    device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

    assert device == "cuda", "#################### CUDA device not detected ! ####################"

    print(f"#################### CUDA device detected ! ####################")

    train_dl, val_dl, class_weights = get_dataloaders(collate_fn = padding_collate, embed_type = "residue", batch_size_train = 1024)

    model = MHAP_MLP(input_embed_dim=1024, output_embed_dim=512, num_classes=5)

    params = {

        "train_dl" : train_dl,
        "val_dl" : val_dl,
        "model" : model,
        "loss_fn" : nn.CrossEntropyLoss(class_weights).to(device),
        "optimizer" : torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2),
        "epochs" : 100,
        "scheduler" : None,
        "logging" : True,
        "grad_accum_steps" : 1, # Just don't
        "output_path" : "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/models/MHAP_MLP_512_out_256_hidden.pth",
        "debug" : False,
        "use_amp" : False

    }

    wandb_project = "MHAP MLP"
    
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