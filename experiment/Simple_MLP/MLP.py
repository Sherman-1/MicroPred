import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

from MHA import MLP, INT_TO_CLASS
from data_utils import EmbedProtT5Dataset

import polars as pl 
import torch 
from torch.utils.data import DataLoader

device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

train_set = torch.load("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/trainset_protein_embeddings.pt", map_location = "cpu")
test_set = torch.load("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/testset_protein_embeddings.pt", map_location = "cpu")

ds = EmbedProtT5Dataset(ptfile="/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/testset_protein_embeddings.pt",
                        num_classes = 5)
dl = DataLoader(ds, batch_size = 10, shuffle = True)
class_weight_dict = ds.class_weights 

input_shape = next(iter(dl))[0].shape[1] 
output_shape = len(class_weight_dict)

print(f"={input_shape}")
print(f"={output_shape}")


model = MLP(input_shape, output_shape).to(device)

for batch in dl: 

    embedding, one_hot, class_type, name = batch
    embedding, one_hot = embedding.to(device), one_hot.to(device)
    print(embedding.shape)
    print(one_hot)
    print(class_type)
    print(name)

    
    model.forward(embedding)
    break


if __name__ == "__main__":

    pass

