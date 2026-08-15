import os
import json
import torch
from datetime import datetime
import torch.nn as nn
from pathlib import Path
from expander_model.expander_model import QueryExpander
from expander_model.dataset import build_dataloaders
from expander_model.config import (train_config, model_config, tokenizer_config, get_tokenizer_class, RUNS_DIR)

num_cores = os.cpu_count()
torch.set_num_threads(max(1, num_cores - 1))
print(f"CPU threads available: {num_cores}")
print(f"Using: {torch.get_num_threads()} threads")

class EarlyStopping:
    """Calls .step(val_loss) once per epoch; returns True once training should stop."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def make_run_dir() -> Path:
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run_name = f"{timestamp}_embed{model_config.embed_dim}_{tokenizer_config.name}"
    run_dir = Path(RUNS_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    return run_dir

def build_optimzer(model: QueryExpander):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    opimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},],
        lr = train_config.learning_rate)
    
    return opimizer

def build_scheduler(optimizer, warmup_steps: int):

    warmup_steps = train_config.warmup_steps

    def lr_lambda(current_steps):
        if current_steps < warmup_steps:
            return current_steps / max(1, warmup_steps)
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model: QueryExpander, loader: torch.utils.data.Dataloader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, criterion: nn.Module, device: torch.device, epoch: int) -> float:
    
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(loader):
        src = batch["src"].to(device)
        tar_in = batch["tar_in"].to(device)
        tar_out = batch["tar_out"].to(device)

        # forward pass
        logits = model(src, tar_in)
        
        batch_size, tar_len, vocab_size = logits.shape

        loss = criterion(logits.view(batch_size * tar_len, vocab_size),
                         tar_out.view(batch_size * tar_len))
        
        # backward pass
        optimizer.zero_grad()                   # clear gradients from prev. step
        loss.backward()

        # clip gradient to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()                       # update weights
        scheduler.step()                       # update lr

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % train_config.log_every == 0:
            avg = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}   |   Step {step+1}  |    Loss {avg:.4f}   |   LR {lr:.6f} ")

    return total_loss / num_batches
    
@torch.no_grad()
def evaluate(model: QueryExpander, loader: torch.utils.data.Dataloader, criterion: nn.Module, device: torch.device) -> float:
    
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        src = batch["src"].to(device)
        tar_in = batch["tar_in"].to(device)
        tar_out = batch["tar_out"].to(device)

        # forward pass
        logits = model(src, tar_in)
        
        batch_size, tar_len, vocab_size = logits.shape

        loss = criterion(logits.view(batch_size * tar_len, vocab_size),
                        tar_out.view(batch_size * tar_len))
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def save_checkpoint(model, tokenizer, epoch, val_loss, run_dir, is_best=False):

    checkpoint = {"epoch": epoch, "val_loss": val_loss, "model_state": model.state_dict()}
    torch.save(checkpoint, run_dir / "checkpoint_latest.pt")

    if is_best:
        torch.save(checkpoint, run_dir / "checkpoint_best.pt")
        print(f"New best model saved (val_loss = {val_loss:.4f})")

    tokenizer.save(str(run_dir / "tokenizer.json"))

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = make_run_dir()
    print(f"Tokenizer: {tokenizer_config.name}")

    TokenizerClass = get_tokenizer_class()
    train_loader, val_loader, tokenizer = build_dataloaders(TokenizerClass)
    tokenizer.save(str(run_dir/"tokenizer.json"))
    model = QueryExpander().to(device)

    print(f"Parameters: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id, label_smoothing=0.1)

    num_training_steps = len(train_loader) * train_config.epochs
    optimizer = build_optimzer(model)
    scheduler = build_scheduler(optimizer, num_training_steps)

    best_val_loss = float("inf")
    history = []
    early_stopping = EarlyStopping(patience=train_config.early_stopping_patience)

    print(f"\nTraining for {train_config.epochs} epochs...\n")
    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % train_config.save_every == 0 or is_best:
            save_checkpoint(model, tokenizer, epoch, val_loss, run_dir, is_best)

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch} — no improvement for {early_stopping.patience} epochs")
            break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {run_dir / 'history.json'}")

    run_metadata = {
        "tokenizer": tokenizer_config.name,
        "data": {
            "total_pairs": len(train_loader.dataset) + len(val_loader.dataset),
            "train_pairs": len(train_loader.dataset),
            "val_pairs":   len(val_loader.dataset),
        },
        "model": {
            "embed_dim":  model_config.embed_dim,
            "num_heads":  model_config.num_heads,
            "num_layers": model_config.num_layers,
            "ff_dim":     model_config.ff_dim,
            "dropout":    model_config.dropout,
        },
        "train": {
            "epochs":        train_config.epochs,
            "batch_size":     train_config.batch_size,
            "learning_rate":  train_config.learning_rate,
        },
        "best_val_loss": best_val_loss,
    }
    with open(run_dir / "config_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    print("\nTesting trained model:")
    test_queries = [
        "a film that makes you think",
        "something dark and unsettling",
        "a feel-good comedy",
    ]
    for query in test_queries:
        expanded = model.generate(query, tokenizer)
        print(f"\nQuery:    {query}")
        print(f"Expanded: {expanded}")

if __name__=="__main__":
    train()

    


