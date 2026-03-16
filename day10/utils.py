import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import sentencepiece as spm


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


def estimate_training_memory(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    precision: str = "fp16",
) -> Dict[str, float]:
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    if precision == "fp16":
        param_memory = param_memory / 2
    
    param_memory_mb = param_memory / (1024 ** 2)
    
    hidden_dim = getattr(model, "args", None)
    if hidden_dim:
        hidden_dim = hidden_dim.dim
    else:
        hidden_dim = 288
    
    activation_memory = batch_size * seq_len * hidden_dim * 4
    if precision == "fp16":
        activation_memory = activation_memory / 2
    activation_memory_mb = activation_memory / (1024 ** 2)
    
    gradient_memory_mb = param_memory_mb
    
    optimizer_memory_mb = param_memory_mb * 2
    
    total_memory_mb = param_memory_mb + activation_memory_mb + gradient_memory_mb + optimizer_memory_mb
    
    return {
        "params_mb": param_memory_mb,
        "activations_mb": activation_memory_mb,
        "gradients_mb": gradient_memory_mb,
        "optimizer_mb": optimizer_memory_mb,
        "total_mb": total_memory_mb,
        "total_gb": total_memory_mb / 1024,
    }


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(num: int) -> str:
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def create_sample_data(
    output_dir: str,
    num_files: int = 1,
    sentences_per_file: int = 100,
    vocab_size: int = 32000,
):
    os.makedirs(output_dir, exist_ok=True)
    
    templates = [
        "Once upon a time, there was a {adj} {noun} who lived in a {place}.",
        "The {adj} {noun} liked to {verb} every day.",
        "One day, the {noun} found a {adj} {thing} in the {place}.",
        "The {noun} was very {emotion} about this discovery.",
        "From that day on, the {noun} always remembered to {verb}.",
    ]
    
    adjectives = ["little", "big", "happy", "sad", "brave", "kind", "old", "young"]
    nouns = ["girl", "boy", "cat", "dog", "bird", "rabbit", "bear", "fox"]
    places = ["forest", "garden", "house", "castle", "mountain", "river", "village"]
    verbs = ["play", "sing", "dance", "run", "jump", "swim", "read", "write"]
    things = ["flower", "treasure", "book", "toy", "friend", "gift", "secret"]
    emotions = ["happy", "excited", "surprised", "curious", "proud", "thankful"]
    
    import random
    
    for file_idx in range(num_files):
        sentences = []
        for _ in range(sentences_per_file):
            template = random.choice(templates)
            sentence = template.format(
                adj=random.choice(adjectives),
                noun=random.choice(nouns),
                place=random.choice(places),
                verb=random.choice(verbs),
                thing=random.choice(things),
                emotion=random.choice(emotions),
            )
            sentences.append(sentence)
        
        text = " ".join(sentences)
        
        output_path = os.path.join(output_dir, f"sample_{file_idx}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"Created {output_path} with {len(sentences)} sentences")


def tokenize_file(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    tokens_array = np.array(tokens, dtype=np.uint16)
    tokens_array.tofile(output_path)
    
    print(f"Tokenized {input_path} -> {output_path}")
    print(f"  Total tokens: {len(tokens)}")


def detokenize_file(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    tokens = np.fromfile(input_path, dtype=np.uint16)
    
    text = tokenizer.decode(tokens.tolist())
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Detokenized {input_path} -> {output_path}")
    print(f"  Total tokens: {len(tokens)}")


def merge_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None,
):
    if weights is None:
        weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
    
    assert len(weights) == len(checkpoint_paths), "Number of weights must match number of checkpoints"
    
    merged_state_dict = None
    
    for path, weight in zip(checkpoint_paths, weights):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        if merged_state_dict is None:
            merged_state_dict = {k: v.clone() * weight for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                merged_state_dict[k] += v * weight
    
    torch.save({"model_state_dict": merged_state_dict}, output_path)
    print(f"Merged {len(checkpoint_paths)} checkpoints -> {output_path}")


def print_model_info(model: torch.nn.Module):
    params = count_parameters(model)
    
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Total parameters: {format_number(params['total'])} ({params['total_M']:.2f}M)")
    print(f"Trainable parameters: {format_number(params['trainable'])} ({params['trainable_M']:.2f}M)")
    
    if hasattr(model, "args"):
        args = model.args
        print(f"\nArchitecture:")
        print(f"  Hidden dim: {args.dim}")
        print(f"  Layers: {args.n_layers}")
        print(f"  Heads: {args.n_heads}")
        print(f"  Max sequence length: {args.max_seq_len}")
        print(f"  Vocabulary size: {args.vocab_size}")
    
    print("=" * 50 + "\n")


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


if __name__ == "__main__":
    print("Utils module test")
    
    sample_dir = "./sample_data"
    create_sample_data(sample_dir, num_files=2, sentences_per_file=50)
    
    print(f"\nSample data created in {sample_dir}")
