import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List
import sentencepiece as spm


class TextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 256,
        cache_dir: Optional[str] = None,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        self.data = self._load_data(data_path, cache_dir)
        
    def _load_data(self, data_path: str, cache_dir: Optional[str] = None) -> np.ndarray:
        if cache_dir and os.path.exists(os.path.join(cache_dir, "data.bin")):
            print(f"Loading cached data from {cache_dir}")
            data = np.fromfile(os.path.join(cache_dir, "data.bin"), dtype=np.uint16)
            return data
        
        all_tokens = []
        
        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                if os.path.isfile(filepath) and filename.endswith('.txt'):
                    print(f"Processing {filepath}")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    tokens = self.tokenizer.encode(text)
                    all_tokens.extend(tokens)
        
        data = np.array(all_tokens, dtype=np.uint16)
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            data.tofile(os.path.join(cache_dir, "data.bin"))
            print(f"Cached data saved to {cache_dir}")
        
        return data
    
    def __len__(self) -> int:
        return (len(self.data) - self.max_seq_len) // self.max_seq_len
    
    def __getitem__(self, idx: int) -> tuple:
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len + 1
        
        chunk = self.data[start_idx:end_idx]
        
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y


class StreamingDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        self.data_files = self._get_data_files(data_path)
        self.current_file_idx = 0
        self.current_tokens = []
        self._load_next_file()
        
    def _get_data_files(self, data_path: str) -> List[str]:
        if os.path.isfile(data_path):
            return [data_path]
        elif os.path.isdir(data_path):
            files = []
            for filename in os.listdir(data_path):
                if filename.endswith('.txt'):
                    files.append(os.path.join(data_path, filename))
            return sorted(files)
        return []
    
    def _load_next_file(self):
        if self.current_file_idx < len(self.data_files):
            filepath = self.data_files[self.current_file_idx]
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            self.current_tokens = self.tokenizer.encode(text)
            self.current_file_idx += 1
    
    def __len__(self) -> int:
        return len(self.current_tokens) // self.max_seq_len
    
    def __getitem__(self, idx: int) -> tuple:
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len + 1
        
        chunk = self.current_tokens[start_idx:end_idx]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_seq_len: int = 256,
        split: str = "train",
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        bin_path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(bin_path):
            self.data = np.fromfile(bin_path, dtype=np.uint16)
        else:
            self.data = self._process_tinystories(data_dir, split, bin_path)
    
    def _process_tinystories(self, data_dir: str, split: str, bin_path: str) -> np.ndarray:
        import json
        
        all_tokens = []
        json_path = os.path.join(data_dir, f"TinyStories_{split}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if isinstance(item, dict) and 'story' in item:
                    text = item['story']
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                
                tokens = self.tokenizer.encode(text)
                all_tokens.extend(tokens)
        
        data_array = np.array(all_tokens, dtype=np.uint16)
        data_array.tofile(bin_path)
        
        return data_array
    
    def __len__(self) -> int:
        return max(1, (len(self.data) - self.max_seq_len) // self.max_seq_len)
    
    def __getitem__(self, idx: int) -> tuple:
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len + 1
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.max_seq_len - 1)
        
        chunk = self.data[start_idx:end_idx]
        
        if len(chunk) < self.max_seq_len + 1:
            chunk = np.pad(chunk, (0, self.max_seq_len + 1 - len(chunk)), constant_values=0)
        
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y


def create_dataloader(
    data_path: str,
    tokenizer_path: str,
    batch_size: int = 32,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = TextDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader


def create_synthetic_data(
    vocab_size: int = 32000,
    seq_len: int = 256,
    num_samples: int = 10000,
    save_path: Optional[str] = None,
) -> np.ndarray:
    data = np.random.randint(0, vocab_size, (num_samples * seq_len,), dtype=np.uint16)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.tofile(save_path)
    
    return data


if __name__ == "__main__":
    print("Dataset module test")
    
    vocab_size = 32000
    seq_len = 256
    num_samples = 1000
    
    synthetic_data = create_synthetic_data(vocab_size, seq_len, num_samples)
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Total tokens: {len(synthetic_data)}")
