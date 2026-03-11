import torch
from torch.utils.data import Dataset
import numpy as np

class LlamaDataset(Dataset):
    def __init__(self, bin_file, max_seq_len):
        self.max_seq_len = max_seq_len
        # 使用内存映射加载二进制文件，不占内存
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        
    def __len__(self):
        # 减去 max_seq_len 是为了保证最后一次取样不越界
        return len(self.data) - self.max_seq_len - 1

    def __getitem__(self, index):
        # 1. 连续取出长度为 max_seq_len + 1 的片段
        chunk = self.data[index : index + self.max_seq_len + 1]
        chunk = chunk.astype(np.int64) # 转为 PyTorch 需要的 int64
        
        # 2. 构造训练对
        # x: 输入序列 [0, 1, 2, ..., n-1]
        # y: 预测序列 [1, 2, 3, ..., n] (即 x 右移一位)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        return x, y

# --- 实验验证 ---
dataset = LlamaDataset("train.bin", max_seq_len=128)
x, y = dataset[0]

print(f"输入 x 的前 5 个 ID: {x[:5]}")
print(f"标签 y 的前 5 个 ID: {y[:5]}")
print(f"x 和 y 的形状是否一致: {x.shape == y.shape}")
