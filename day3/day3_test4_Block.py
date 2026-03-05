import torch
import torch.nn as nn
import numpy as np

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 典型的升维 -> 激活 -> 降维 结构
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class AddNorm(nn.Module):
    """
    实现：LayerNorm(x + Sublayer(x))
    注意：这里我们采用 Pre-LN 结构，即：x + Sublayer(LayerNorm(x))
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        # 1. 先做 Norm (Pre-LN)
        norm_x = self.norm(x)
        # 2. 经过子层计算 (Attention 或 FFN)
        sub_output = sublayer_fn(norm_x)
        # 3. 残差连接：原输入 x + 经过处理后的输出
        return x + self.dropout(sub_output)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        from day2.day2_test2_mha import MultiHeadAttention # 假设你保存了昨天的代码
        
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        # 第一步：自注意力 + 残差归一化
        x = self.add_norm1(x, lambda x: self.attn(x, x, x, mask)[0])
        # 第二步：前馈网络 + 残差归一化
        x = self.add_norm2(x, self.ff)
        return x
