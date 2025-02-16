#!/usr/bin/env python3
"""
Toy example for debugging Trainer.train() with a REG_CLASS model and toy datasets.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys, os
os.environ["TRITON_CACHE_DIR"] = "/scratchlocal/triton_cache"

# ---------------------------
# Minimal MLP module
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------
# REG_CLASS model
# ---------------------------
class REG_CLASS(nn.Module):
    def __init__(self, input_embed_dim, hidden_dim, num_classes, descriptors_dim, device):
        super(REG_CLASS, self).__init__()
        self.device = device
        self.classifier = MLP(input_dim=input_embed_dim, hidden_dim=hidden_dim, output_dim=num_classes).to(device)
        self.regressor = MLP(input_dim=input_embed_dim, hidden_dim=hidden_dim, output_dim=descriptors_dim).to(device)
        # BCEWithLogitsLoss expects one-hot targets (float) matching the logits shape.
        self.classif_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.MSELoss()
    
    def forward(self, embeddings, labels=None, phychem_descriptors=None):

        embeddings = embeddings.to(self.device)
        class_output = self.classifier(embeddings)
        reg_output = self.regressor(embeddings)
        
        if labels is not None and phychem_descriptors is not None:
            labels = labels.to(self.device)
            phychem_descriptors = phychem_descriptors.to(self.device)
            loss_class = self.classif_loss_fn(class_output, labels)
            loss_reg = self.reg_loss_fn(reg_output, phychem_descriptors)
            combined_loss = loss_class + loss_reg
            return {"loss": combined_loss, 
                    "logits": class_output
                    }
        else:
            return {"logits": class_output}

# ---------------------------
# Create toy datasets
# ---------------------------
def create_toy_dataset(num_samples=100, input_dim=10, num_classes=3, descriptors_dim=5):

    # Random embeddings: shape (num_samples, input_dim)
    embeddings = np.random.randn(num_samples, input_dim).astype(np.float32)
    # Random one-hot labels for classification: shape (num_samples, num_classes)
    class_indices = np.random.randint(0, num_classes, size=(num_samples,))
    labels = np.eye(num_classes)[class_indices].astype(np.float32)
    # Random regression targets: shape (num_samples, descriptors_dim)
    phychem_descriptors = np.random.randn(num_samples, descriptors_dim).astype(np.float32)
    
    data = {
        "embeddings": embeddings,
        "labels": labels,
        "phychem_descriptors": phychem_descriptors,
    }
    return Dataset.from_dict(data)

train_dataset = create_toy_dataset(num_samples=100)
eval_dataset = create_toy_dataset(num_samples=20)

# ---------------------------
# Custom compute_metrics function using evaluate.load() style (Option 2)
# ---------------------------
def compute_metrics(eval_pred):

    # Now eval_pred.predictions should be a single numpy array
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(labels, axis=-1)
    # Compute your metrics...
    acc = accuracy_score(true_classes, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, preds, average="macro")
    return {"eval_accuracy": acc, "eval_precision": precision, "eval_recall": recall, "eval_f1": f1}

# ---------------------------
# Setup Trainer
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = REG_CLASS(input_embed_dim=10, hidden_dim=16, num_classes=3, descriptors_dim=5, device=device)

training_args = TrainingArguments(
    output_dir="./toy_model",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to=[],  # Disable logging
    greater_is_better=True,
    remove_unused_columns=False,  # Ensure all columns are passed to the model
    prediction_loss_only=False,
    disable_tqdm=True   
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# ---------------------------
# Debug: Run training
# ---------------------------
if __name__ == "__main__":

    print()
    # Optionally, run an initial evaluation to inspect output
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
    # Run training; watch the logs to see if compute_metrics is called
    trainer.train()
