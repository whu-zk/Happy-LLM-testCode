import torch.nn.functional as F
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        
        # 1. 计算 LLaMA 特有的隐藏层维度 (8/3 * d_model 并对齐)
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 2. 定义三个线性层 (注意：LLaMA 通常不带 bias)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # 对应公式：(Swish(xW1) * xW3) * W2
        # F.silu 就是 Swish 激活函数
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
