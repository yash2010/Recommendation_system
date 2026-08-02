import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from model.model import QueryExpander
from model.dataset import build_dataloaders
from model.tokenizer import Tokenizer
from model.config import train_config, model_config

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

def build_scheduler(optimizer, num_training_step: int):

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

def save_checkpoint(model: QueryExpander, tokenizer: Tokenizer, epoch: int, val_loss: float, save_dir: str, is_best: bool = False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)

    checkpoint = {"epoch": epoch, "val_loss": val_loss, "model_state": model.state_dict()}
    torch.save(checkpoint, save_dir / "checkpoint_latest.pt")

    if is_best:
        torch.save(checkpoint, save_dir / "checkpoint_best.pt")
        print(f"New best model saved (val_loss = {val_loss:.4f})")

    tokenizer.save(str(save_dir / "tokenizer.json"))

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, tokenizer = build_dataloaders()

    model = QueryExpander().to(device)

    print(f"Parameters: {model.count_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id, label_smoothing=0.1)

    num_training_steps = len(train_loader) * train_config.epochs
    optimizer = build_optimzer(model)
    scheduler = build_scheduler(optimizer, num_training_steps)

    best_val_loss = float("inf")
    history = []

    print(f"\nTraining for {train_config.epochs} epochs...\n")
    for epoch in range(1, train_config.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if epoch % train_config.save_every == 0 or is_best:
            save_checkpoint(model, tokenizer, epoch, val_loss, train_config.save_dir, is_best)
        
        with open(Path(train_config.save_dir) / "history", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

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

    


