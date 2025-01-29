
import random
import numpy as np 
import polars as pl 

from datasets import Dataset

import torch

import torch.nn as nn

from transformers import T5Tokenizer, T5EncoderModel, TrainingArguments, Trainer
from peft import  get_peft_model, LoraConfig, TaskType


if not torch.cuda.is_available():

    exit("Ola oh tu fais quoi là !!!")

class ProtT5Classifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Linear(1024, num_classes, device = "cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1) 
        logits = self.classifier(pooled_output)
        return logits
    

def create_dataset(tokenizer, seqs, labels):

    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True) 
    tokenized["labels"] = labels
    
    return Dataset.from_dict(tokenized)

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


 
data = pl.scan_csv("../splits/three_vs_rest.csv").with_columns(pl.when(pl.col("target") >= 1).then(pl.lit(1)).otherwise(pl.lit(0)).alias("label"))

train_data = data.filter(pl.col("set") == "train").select(["sequence","label"]).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String)
).collect().sample(200)

test_data = data.filter(pl.col("set") == "test").select(["sequence","label"]).with_columns(
    sequence = pl.col("sequence").str.replace(r"U|B|O|Z","X").replace("*", "").map_elements(lambda seq : " ".join(seq), return_dtype = pl.String)
).collect().sample(200)


train_dataset = create_dataset(tokenizer = tokenizer,
                               seqs = train_data.select("sequence").to_series().to_list(),
                               labels = train_data.select("label").to_series().to_list()
)

test_dataset = create_dataset(tokenizer = tokenizer,
                               seqs = test_data.select("sequence").to_series().to_list(),
                               labels = test_data.select("label").to_series().to_list()
)


lora_config = LoraConfig(
    r=4,  # Rank of LoRA matrices (trade-off between efficiency & quality)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for regularization
    target_modules=["q", "v"],  # Apply LoRA to query (`q`) and value (`v`) layers
    bias="none",  # No bias update
    task_type=TaskType.FEATURE_EXTRACTION  # Since we're using an encoder model
)

base_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float32).to(torch.device('cuda'))
peft_model = get_peft_model(base_model, lora_config)
model = ProtT5Classifier(peft_model, num_classes=2)



# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Sample test sequences
test_sequences = ["MKVLLFAIPTAAAVVTLATGNPQKTVTIKTG", "MNGTEGPNFYVPFSNKTGVVRSPFEYPQYYLA"]
test_labels = torch.tensor([0, 1], dtype=torch.long).to(device)  # Binary classification labels

# Tokenize the test sequences
tokenized_inputs = tokenizer(
    test_sequences, max_length=1024, padding=True, truncation=True, return_tensors="pt"
)

# Move inputs to the correct device
tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

# Define a loss function (CrossEntropy for classification)
criterion = torch.nn.CrossEntropyLoss()

# Forward pass to compute logits
logits = model(**tokenized_inputs)

# Compute loss
loss = criterion(logits, test_labels)

# Backward pass to compute gradients
loss.backward()

# Print loss
print("\n✅ Backward pass successful!")
print("Loss:", loss.item())

# Check which parameters received gradients
print("\n📌 Checking gradients:")

for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"✔ {name} has gradients! Shape: {param.grad.shape}")
    elif param.requires_grad and param.grad is None:
        print(f"❌ {name} has NO gradients!")

# Optional: Zero gradients before next step (common practice in training)
model.zero_grad()


