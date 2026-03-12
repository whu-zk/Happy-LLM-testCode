import torch
import torch.nn as nn
from day9_test1_RMSNorm import RMSNorm
from day9_test4_Attention import Attention
from day9_test2_FeedForward import FeedForward
from day9_test5_TransformerBlock import TransformerBlock

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # dim: 每个头的维度 (head_dim)
    # end: 最大长度 (max_seq_len)
    
    # 1. 计算频率 theta_i
    # shape: (dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. 生成位置 m [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)  
    
    # 3. 外积计算 m * theta_i
    # shape: (end, dim // 2)
    freqs = torch.outer(t, freqs).float()
    
    # 4. 转化为复数形式: cos(m*theta) + i*sin(m*theta)
    # shape: (end, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

class Llama(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 1. 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # 2. 堆叠 Transformer Blocks
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

        # 3. 输出前的归一化和线性层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 4. 预计算 RoPE 频率矩阵
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 2
        )

    def forward(self, tokens, start_pos):
        # tokens: [batch, seq_len]
        _batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens) # [batch, seq_len, dim]

        # 获取对应的 RoPE 频率
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        # 构造因果掩码 (Causal Mask)
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

        # 逐层通过 Transformer Blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        # 最终归一化和输出投影
        h = self.norm(h)
        output = self.output(h).float() # [batch, seq_len, vocab_size]

        return output
