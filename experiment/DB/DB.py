import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MLP

import polars as pl 
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

class CLASSIF_REG(nn.Module):

    def __init__(self, input_embed_dim: int, hidden_dim: int, num_classes: int, descriptors_dim: int, class_weights, device):
        super(CLASSIF_REG, self).__init__()
        self.classifier = MLP(input_shape=input_embed_dim, hidden_dim=hidden_dim, output_shape=num_classes)
        self.regressor = MLP(input_shape=input_embed_dim, hidden_dim=hidden_dim, output_shape=descriptors_dim)
        self.classif_loss_fn = nn.BCEWithLogitsLoss(weight = torch.as_tensor(class_weights, dtype = torch.float32, device = device))
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, embeddings, mask=None, labels_class=None, labels_reg=None):
        class_output = self.classifier(embeddings)
        reg_output = self.regressor(embeddings)
        loss = None
        if labels_class is not None and labels_reg is not None:
            loss_class =  self.classif_loss_fn(class_output, labels_class)
            loss_reg = self.reg_loss_fn(reg_output, labels_reg)

            loss = loss_class + loss_reg

        if loss is not None:
            return {"loss": loss, "class_logits": class_output, "reg_output": reg_output}
        else:
            return class_output, reg_output

def main():

    pass
    
    
if __name__ == "__main__":

    main()