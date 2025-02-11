
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
from transformers import TrainingArguments, Trainer
from peft import  get_peft_model, LoraConfig, TaskType
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

import sys, os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent  
src_path = root_dir / 'src'
sys.path.append(str(src_path))

os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

from MHA import MLP
from utils import print_gpu_memory, check_model_on_gpu, Trainer as CustomTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not torch.cuda.is_available() : exit("Cuda compatible GPU not found")

class ProtT5Classifier(nn.Module):
    def __init__(self, base_model, num_classes, loss_weights=None):
        super().__init__()
        self.encoder = base_model
        self.classifier = MLP(input_dim=1024, hidden_dim=512, output_dim=num_classes)
        # Create your loss function here. If loss_weights is provided, use it.
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(loss_weights, device=base_model.device, dtype=torch.float32)) \
            if loss_weights is not None else nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else logits

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

    TOKENIZER = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


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

    train_ds = TensorDataset(
        tokenized["input_ids"], 
        train_data["category"].to_torch().long(),
        F.one_hot(train_data["category"].to_torch(), num_classes=5).float(),
        tokenized["attention_mask"],
        torch.zeros(train_data["category"].to_torch().shape[0]) # Fake sequence ID just for the DataLoader
    )

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

    val_ds = TensorDataset(
        tokenized["input_ids"], 
        val_data["category"].to_torch().long(),
        F.one_hot(val_data["category"].to_torch(), num_classes=5).float(),
        tokenized["attention_mask"],
        torch.zeros(val_data["category"].to_torch().shape[0])

    )

    def data_collator(features):
        """
        Convert a list of tuples into a dict with the keys that match your model's forward() arguments.
        Expected tuple order: (input_ids, category, one_hot, attention_mask, fake_id)
        We only need 'input_ids', 'attention_mask', and use 'category' as labels.
        """
        input_ids = torch.stack([f[0] for f in features])
        attention_mask = torch.stack([f[3] for f in features])
        labels = torch.stack([f[1] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        accuracy = (preds == torch.tensor(labels)).float().mean().item()
        return {"accuracy": accuracy}
    
    train_labels = train_ds.tensors[1].tolist()
    classes = np.unique(train_labels)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    cw = torch.tensor(cw, dtype=torch.float32, device=DEVICE)


    BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(DEVICE)
    PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
    MODEL = ProtT5Classifier(PEFT_MODEL, 5, loss_weights = cw).to(DEVICE)
    check_model_on_gpu(MODEL)


    training_args = TrainingArguments(
        output_dir='/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_MLP/results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=126,
        eval_strategy="epoch",
        fp16=True,  
        deepspeed='/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/experiment/FT_MLP/ds_config.json',  
        save_strategy="epoch",
        logging_steps=50,
        report_to=["wandb"],  
        run_name="FT_DeepSpeed" 
    )

    trainer = Trainer(
        model=MODEL,
        args=training_args,
        train_dataset=train_ds,   
        eval_dataset=val_ds,      
        data_collator=data_collator,     
        compute_metrics=compute_metrics  
    )

    wandb.init(project="FT-MLP", name="FT_DeepSpeed")

    trainer.train()

if __name__ == "__main__": 

    main()