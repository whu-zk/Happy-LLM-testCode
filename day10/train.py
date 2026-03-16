import os
import json
import math
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import Llama, ModelArgs, count_parameters
from dataset import create_dataloader, create_synthetic_data


class CosineAnnealingWithWarmup:
    def __init__(
        self,
        learning_rate: float,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: float,
    ):
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def get_lr(self, it: int) -> float:
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        
        if it > self.lr_decay_iters:
            return self.min_lr
        
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        model_config = ModelArgs(**config["model"])
        self.model = Llama(model_config).to(device)
        
        params_info = count_parameters(self.model)
        print(f"Model Parameters: {params_info['total_params_M']:.2f}M")
        
        self.optimizer = self._create_optimizer()
        self.scheduler = CosineAnnealingWithWarmup(
            learning_rate=config["training"]["learning_rate"],
            warmup_iters=config["training"]["warmup_iters"],
            lr_decay_iters=config["training"]["lr_decay_iters"],
            min_lr=config["training"]["min_lr"],
        )
        
        self.scaler = GradScaler()
        self.grad_clip = config["training"]["grad_clip"]
        
        self.current_iter = 0
        self.best_loss = float("inf")
        
        self.checkpoint_dir = config["checkpoint"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.writer = SummaryWriter(log_dir)
        
        if use_wandb:
            import wandb
            wandb.init(
                project="tiny-llm",
                config=config,
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        weight_decay = self.config["training"]["weight_decay"]
        
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'norm' in name.lower() or 'bias' in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config["training"]["learning_rate"],
            betas=(0.9, 0.95),
        )
        
        return optimizer
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["iter"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"Loaded checkpoint from iteration {self.current_iter}")
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            "iter": self.current_iter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{self.current_iter}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train_step(self, batch: tuple) -> float:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        
        with autocast():
            logits = self.model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
        
        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip,
        )
        
        lr = self.scheduler.get_lr(self.current_iter)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train(
        self,
        dataloader,
        max_iters: int,
        save_interval: int = 500,
        log_interval: int = 10,
        eval_interval: int = 100,
        eval_dataloader=None,
    ):
        self.model.train()
        
        data_iter = iter(dataloader)
        losses = []
        start_time = time.time()
        
        pbar = tqdm(range(max_iters), desc="Training")
        
        for iteration in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            loss = self.train_step(batch)
            losses.append(loss)
            self.current_iter += 1
            
            if iteration % log_interval == 0:
                avg_loss = sum(losses[-log_interval:]) / len(losses[-log_interval:])
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config["training"]["batch_size"]
                    * self.config["model"]["max_seq_len"]
                    * log_interval
                    / elapsed
                )
                
                lr = self.scheduler.get_lr(self.current_iter)
                
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_sec=f"{tokens_per_sec:.0f}",
                )
                
                self.writer.add_scalar("train/loss", avg_loss, self.current_iter)
                self.writer.add_scalar("train/learning_rate", lr, self.current_iter)
                self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.current_iter)
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/iteration": self.current_iter,
                    })
                
                start_time = time.time()
            
            if eval_dataloader and iteration % eval_interval == 0 and iteration > 0:
                eval_loss = self.evaluate(eval_dataloader)
                self.writer.add_scalar("eval/loss", eval_loss, self.current_iter)
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "eval/loss": eval_loss,
                        "train/iteration": self.current_iter,
                    })
                
                is_best = eval_loss < self.best_loss
                if is_best:
                    self.best_loss = eval_loss
                
                self.save_checkpoint(is_best)
                self.model.train()
            
            if iteration % save_interval == 0 and iteration > 0:
                self.save_checkpoint()
        
        self.save_checkpoint()
        self.writer.close()
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        print("Training completed!")
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            with autocast():
                logits = self.model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:
                break
        
        avg_loss = total_loss / num_batches
        print(f"\nEvaluation loss: {avg_loss:.4f}")
        return avg_loss


def estimate_memory(model, batch_size, seq_len, device="cuda"):
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    param_memory_mb = param_memory / (1024 ** 2)
    
    activation_memory = batch_size * seq_len * model.args.dim * 4
    activation_memory_mb = activation_memory / (1024 ** 2)
    
    gradient_memory = param_memory_mb
    
    optimizer_memory = param_memory_mb * 2
    
    total_memory = param_memory_mb + activation_memory_mb + gradient_memory + optimizer_memory
    
    print(f"\nMemory Estimation:")
    print(f"  Parameters: {param_memory_mb:.2f} MB")
    print(f"  Activations: {activation_memory_mb:.2f} MB")
    print(f"  Gradients: {gradient_memory:.2f} MB")
    print(f"  Optimizer states: {optimizer_memory:.2f} MB")
    print(f"  Total (estimated): {total_memory:.2f} MB")
    
    return total_memory


def main():
    parser = argparse.ArgumentParser(description="Train a small LLaMA model")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny_config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to training data",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer.model",
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = Trainer(config, device=device, use_wandb=args.use_wandb)
    
    estimate_memory(
        trainer.model,
        config["training"]["batch_size"],
        config["model"]["max_seq_len"],
        device,
    )
    
    if args.synthetic:
        print("\nUsing synthetic data for testing...")
        vocab_size = config["model"]["vocab_size"]
        seq_len = config["model"]["max_seq_len"]
        num_samples = 10000
        
        data = create_synthetic_data(vocab_size, seq_len, num_samples)
        
        from dataset import TextDataset
        import numpy as np
        
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, data, max_seq_len):
                self.data = data
                self.max_seq_len = max_seq_len
            
            def __len__(self):
                return (len(self.data) - self.max_seq_len) // self.max_seq_len
            
            def __getitem__(self, idx):
                start = idx * self.max_seq_len
                end = start + self.max_seq_len + 1
                chunk = self.data[start:end]
                return torch.from_numpy(chunk[:-1].astype(np.int64)), torch.from_numpy(chunk[1:].astype(np.int64))
        
        dataset = SyntheticDataset(data, seq_len)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )
    else:
        dataloader = create_dataloader(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            batch_size=config["training"]["batch_size"],
            max_seq_len=config["model"]["max_seq_len"],
        )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(
        dataloader=dataloader,
        max_iters=config["training"]["max_iters"],
        save_interval=config["training"]["save_interval"],
        log_interval=config["training"]["log_interval"],
        eval_interval=config["training"]["eval_interval"],
    )


if __name__ == "__main__":
    main()
