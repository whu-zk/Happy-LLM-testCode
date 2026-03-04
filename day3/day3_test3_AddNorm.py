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

# --- 维度实验 ---
d_model = 512
x = torch.randn(2, 10, d_model)  # Batch=2, Seq=10, Dim=512
add_norm = AddNorm(d_model)

# 模拟一个子层（比如 FFN）
ff = FeedForward(d_model, d_ff=2048)

# 运行
output = add_norm(x, ff)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}") 
assert x.shape == output.shape, "警告：残差连接前后维度不一致！"
