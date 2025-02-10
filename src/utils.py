import torch 
import numpy as np 
import random 
from transformers import set_seed
import wandb
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report


def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

def print_gpu_memory(msg=""):
    allocated = torch.cuda.memory_allocated() / 1e9  
    reserved = torch.cuda.memory_reserved() / 1e9 
    print(f"\n {msg}")
    print(f"   - Allocated: {allocated:.2f} GB")
    print(f"   - Reserved: {reserved:.2f} GB")

def check_model_on_gpu(model):
    is_on_gpu = all(param.device.type == "cuda" for param in model.parameters())
    if is_on_gpu: 
        print("Model is fully on GPU") 
    else:
        exit("Some parameters are still on CPU")

def compute_sklearn_metrics(model, val_dl, device):
    """Compute and print accuracy metrics on the validation set using sklearn."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dl, start=1):

            if isinstance(batch, (tuple, list)):
                data = batch[0]
                target = batch[1]
                mask = batch[3] if len(batch) > 4 else None # Dirty ikr
            else:
                raise ValueError("Batch format not recognized.")

            data = data.to(device)
            target = target.to(device)
            if mask is not None:
                mask = mask.to(device)
        
            outputs = model(data, mask) if mask is not None else model(data)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:")
    print(classification_report(all_targets, all_preds))
    return accuracy


class Trainer:
    def __init__(
        self,
        model : torch.nn.Module,
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
        grad_accum_steps=1,         
        output_path = None,
        debug = False,
        use_amp = False

    ):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            train_dl (DataLoader): Training dataloader.
            val_dl (DataLoader, optional): Validation dataloader. Defaults to None.
            optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to None.
            loss_fn (callable, optional): Loss function. Defaults to None.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
            device (str, optional): Device to use ("cuda" or "cpu"). Defaults to auto-select.
            wandb_project (str, optional): wandb project name. Defaults to "default_project".
            wandb_config (dict, optional): Additional wandb configuration.
            val_interval_batches (int, optional): Frequency (in batches) to compute and log validation loss during training.
            grad_accum_steps (int, optional): Number of mini-batches over which to accumulate gradients before performing an optimizer step.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.scheduler = scheduler
        self.grad_accum_steps = grad_accum_steps
        self.output_path = output_path
        self.logging = logging
        self.debug = debug
        self.use_amp = use_amp

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        if self.logging:
            if wandb_config is None:
                wandb_config = {}
            wandb.init(project=wandb_project, config=wandb_config)
            wandb.watch(self.model, log="all")

        assert self.device != "cpu", "Model running on CPU ? Not today"

        self.model = self.model.to(device)

    def save_checkpoint(self, path):
        """Saves the model and optimizer states."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def compute_validation(self):
        """Computes the validation loss and accuracy over the entire validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_dl, start=1):
            
                # Without mask : embedding, target, one_hot_target, name
                # With mask : embeddings, target, one_hot, mask, names
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                    target = batch[1]
                    mask = batch[3] if len(batch) > 4 else None # Dirty ikr
                else:
                    raise ValueError("Batch format not recognized.")

                data = data.to(self.device)
                target = target.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)
                outputs = self.model(data, mask) if mask is not None else self.model(data)
                loss = self.loss_fn(outputs, target)
                total_loss += loss.item()

                if outputs.dim() > 1 and outputs.size(1) > 1:
                    preds = outputs.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)

        avg_loss = total_loss / len(self.val_dl)
        accuracy = correct / total if total > 0 else 0.0
        self.model.train()
        return avg_loss, accuracy

    def validate(self):
        """Runs validation at the end of an epoch and logs the results."""
        val_loss, accuracy = self.compute_validation()
        if self.logging:
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": accuracy
            })


    def train(self):
        """Runs the training loop over the specified number of epochs."""
        best_val_loss = float("inf")

        num_batches = len(self.train_dl)
        if self.debug : print(f"{num_batches} batch per epoch !")

        for epoch in tqdm(range(1, self.epochs + 1), desc="Epoch", total=self.epochs, leave=False):
            self.model.train()
            running_loss = 0.0
            self.optimizer.zero_grad()

            if self.debug : print_gpu_memory(f"Epoch {epoch} start")

            for batch_idx, batch in enumerate(self.train_dl, start=1):
            
                # Without mask : embedding, target, one_hot_target, name
                # With mask : embeddings, target, one_hot, mask, names
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                    target = batch[1]
                    mask = batch[3] if len(batch) > 4 else None # Dirty ikr
                else:
                    raise ValueError("Batch format not recognized.")

                data = data.to(self.device)
                target = target.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                if self.debug : print_gpu_memory(f"Epoch {epoch} Batch {batch_idx} after data transfer")

                if self.use_amp:
                    if self.debug : print("Using Autocast ...")
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data, mask) if mask is not None else self.model(data)
                        if self.debug : print(f"Logits data type : {outputs.dtype}"); print(f"Target data type : {target.dtype}")
                        loss = self.loss_fn(outputs, target) / self.grad_accum_steps
                else:
                    outputs = self.model(data, mask) if mask is not None else self.model(data)
                    loss = self.loss_fn(outputs, target) / self.grad_accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if self.debug : print_gpu_memory(f"Epoch {epoch} Batch {batch_idx} after backward")

                running_loss += loss.item() * self.grad_accum_steps

                if batch_idx % self.grad_accum_steps == 0 or batch_idx == len(self.train_dl):
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                    else:
                        self.optimizer.step()

                    if self.debug : print_gpu_memory(f"Epoch {epoch} Batch {batch_idx} after optimizer step")
                    self.optimizer.zero_grad()

            avg_loss = running_loss / len(self.train_dl)
            if self.logging:
                wandb.log({"train_loss": avg_loss, "epoch": epoch})

            if self.scheduler is not None:
                self.scheduler.step()

            if self.val_dl is not None:
                val_loss, accuracy = self.compute_validation()
                if self.logging:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_accuracy": accuracy,
                        "epoch": epoch
                    })

                if val_loss < best_val_loss and self.output_path:
                    best_val_loss = val_loss
                    print(f"New best model found at epoch {epoch} with val loss {val_loss:.4f}. Saving checkpoint.")
                    self.save_checkpoint(self.output_path)

            if self.debug:
                print("Debug mode activated: stopping training early.")
                break



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


    ###############
    # Toy Example 
    ###############

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset


    X = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(X, y)
    train_dl = DataLoader(dataset, batch_size=10, shuffle=True)
    val_dl = DataLoader(dataset, batch_size=10)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        scheduler=scheduler,
        wandb_project="my_wandb_project",
        wandb_config={"learning_rate": 0.001, "batch_size": 16},
        val_interval_batches=10,   
        grad_accum_steps=4         
    )

    trainer.train()
