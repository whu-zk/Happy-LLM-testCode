import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 创建一个足够长的 PE 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成位置序列 [0, 1, 2, ..., max_len-1] 并增加一个维度变成 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算公式中的分母部分 (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 4. 填充 PE 矩阵：偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 增加 batch 维度 (1, max_len, d_model) 并注册为 buffer（不参与梯度下降）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        # 将 PE 加到输入 Embedding 上（只取当前句子长度的部分）
        x = x + self.pe[:, :x.size(1)]
        return x

# --- 可视化实验 ---
pe_model = PositionalEncoding(d_model=128, max_len=100)
plt.figure(figsize=(10, 5))
plt.imshow(pe_model.pe[0].cpu().numpy(), cmap='RdBu')
plt.title("Positional Encoding Heatmap")
plt.xlabel("Dimension (d_model)")
plt.ylabel("Position (pos)")
plt.colorbar()
plt.show()
plt.savefig('./positional_encoding.png')


