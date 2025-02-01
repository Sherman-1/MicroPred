
import random
import numpy as np 
import polars as pl 

from datasets import Dataset as HFDataset

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from transformers import T5Tokenizer, T5EncoderModel, TrainingArguments, Trainer
from peft import  get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn

import wandb

NUM_CLASSES = 5
BATCH_SIZE = 20
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_KEY = "b0b0b462179eb67f29990745021c903e24636abd"
wandb.login(key=WANDB_KEY)

wandb.init(project="ProtT5_Finetuning", 
           config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH
        },
        tags=["dev"]
)

if not torch.cuda.is_available() : exit("Cuda compatible GPU not found")

def print_gpu_memory(msg=""):
    allocated = torch.cuda.memory_allocated() / 1e9  
    reserved = torch.cuda.memory_reserved() / 1e9 
    print(f"\n🔹 {msg}")
    print(f"   - Allocated: {allocated:.2f} GB")
    print(f"   - Reserved: {reserved:.2f} GB")

def check_model_on_gpu(model):
    is_on_gpu = all(param.device.type == "cuda" for param in model.parameters())
    if is_on_gpu: 
        print("✅ Model is fully on GPU") 
    else:
        exit("❌ Some parameters are still on CPU")

def create_dataset(tokenizer, seqs, labels):
    tokenized = tokenizer(seqs, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
    tokenized["labels"] = labels
    return HFDataset.from_dict(tokenized)

class ProtT5Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset  

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        item = {key: torch.tensor(val) for key, val in sample.items()}
        item["labels"] = torch.tensor(sample["labels"], dtype=torch.long)
        return item
    
class ProtT5Classifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),             
            nn.Dropout(0.3),       
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1) 
        logits = self.classifier(pooled_output)
        return logits

torch.cuda.empty_cache()  
torch.cuda.synchronize()  
print("VRAM cleared!")
print_gpu_memory("GPU Memory before loading model :")

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

train_data = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/train_sequences.csv")
test_data = pl.read_csv("/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/training_dataset/test_sequences.csv")


LORA_CONFIG = LoraConfig(
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.1,  
    target_modules=["q", "v"],  
    bias="none",  
    task_type=TaskType.FEATURE_EXTRACTION  
)



hf_train_dataset = create_dataset(tokenizer, train_data["sequence"].to_list(), train_data["category"].to_list())
train_dataset = ProtT5Dataset(hf_train_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

hf_test_dataset = create_dataset(tokenizer, test_data["sequence"].to_list(), test_data["category"].to_list())
test_dataset = ProtT5Dataset(hf_test_dataset)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True) # Since no grad accum ?

BASE_MODEL = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16).to(DEVICE)
PEFT_MODEL = get_peft_model(BASE_MODEL, LORA_CONFIG)
MODEL = ProtT5Classifier(PEFT_MODEL, NUM_CLASSES).to(DEVICE)

check_model_on_gpu(MODEL)
print_gpu_memory("GPU Memory after loading model :")

optimizer = optim.AdamW(MODEL.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

best_val_loss = float("inf")
best_model_path = "models/310125_FT_1LAYERCLASSIF.pth"
for epoch in tqdm(range(EPOCHS), desc = "EPOCH"):

    MODEL.train()
    total_loss = 0.0
    for step, batch in tqdm(enumerate(train_loader), leave=False, desc = "BATCH"):
        
        batch = {key: val.to(DEVICE) for key, val in batch.items()}
        labels = batch.pop("labels")  

        with torch.cuda.amp.autocast():  
            logits = MODEL(**batch)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if (step % GRADIENT_ACCUMULATION_STEPS == GRADIENT_ACCUMULATION_STEPS - 1) or (step == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        wandb.log({
            "train_loss": loss.item(),
            "step": step + (epoch * len(train_loader))  # Total 
        })

    avg_train_loss = total_loss / len(train_loader)
    wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch + 1})

    MODEL.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_step, batch in tqdm(enumerate(test_loader), leave=False, desc="VALIDATION"):
            batch = {key: val.to(DEVICE) for key, val in batch.items()}
            labels = batch.pop("labels")

            with torch.cuda.amp.autocast():
                logits = MODEL(**batch)
                loss = criterion(logits, labels)

            total_val_loss += loss.item()

            wandb.log({
                "val_loss": loss.item(),
                "val_step": val_step + (epoch * len(test_loader))
            })

    avg_val_loss = total_val_loss / len(test_loader)

    wandb.log({"avg_val_loss": avg_val_loss, "epoch": epoch + 1})
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(MODEL.cpu().state_dict(), best_model_path)
    

    torch.cuda.empty_cache()

wandb.finish()  


MODEL.load_state_dict(torch.load(best_model_path))
MODEL.eval()

total_val_loss = 0.0
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, leave=False, desc="FINAL EVALUATION"):
        batch = {key: val.to(DEVICE) for key, val in batch.items()}
        labels = batch.pop("labels")

        with torch.cuda.amp.autocast():
            logits = MODEL(**batch)
            loss = criterion(logits, labels)

        total_val_loss += loss.item()

        # Logits to pred
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).cpu().numpy()
        labels = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

final_val_loss = total_val_loss / len(test_loader)

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

metrics_table = wandb.Table(columns=["Epoch", "Validation Loss", "Accuracy", "Precision", "Recall", "F1 Score"])

metrics_table.add_data(epoch + 1, avg_val_loss, acc, precision, recall, f1)

wandb.log({"Validation Metrics Table": metrics_table})


