import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x 形状: (batch, seq_len, n_kv_heads, head_dim)
    n_rep: 重复次数 (n_heads // n_kv_heads)
    """
    if n_rep == 1:
        return x
    
    # 在维度 2 (头维度) 后面插入一个新维度，进行扩展后展平
    batch, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    )

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x shape: [batch, seq_len, n_heads, head_dim]
    # freqs_cis shape: [seq_len, head_dim // 2]
    
    # 1. 将 x 的最后一个维度配对成复数
    # [batch, seq_len, n_heads, head_dim] -> [batch, seq_len, n_heads, head_dim // 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # 2. 准备 freqs_cis 的维度以便广播
    # [seq_len, head_dim // 2] -> [1, seq_len, 1, head_dim // 2]
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, x_complex.shape[-1])
    
    # 3. 复数乘法即旋转：x * e^(im*theta)
    x_rotated = x_complex * freqs_cis
    
    # 4. 转回实数并恢复形状
    # [batch, seq_len, n_heads, head_dim // 2, 2] -> [batch, seq_len, n_heads, head_dim]
    out = torch.view_as_real(x_rotated).flatten(3)
    return out.type_as(x)

class Attention(nn.Module):
    def __init__(self, args): # args 包含 dim, n_heads, n_kv_heads 等
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Q, K, V 的线性变换
        # 注意：K 和 V 的输出维度是 n_kv_heads * head_dim
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask):
        batch, seq_len, _ = x.shape
        
        # 1. 投影并变形
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(batch, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 应用 RoPE 旋转位置编码
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # 3. GQA 核心：重复 KV 头以匹配 Q 的数量
        keys = repeat_kv(xk, self.n_rep)   # (batch, seq_len, n_heads, head_dim)
        values = repeat_kv(xv, self.n_rep) # (batch, seq_len, n_heads, head_dim)

        # 4. 调整维度用于计算: (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 5. 计算 Scaled Dot-Product Attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # 加上因果掩码
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values) # (batch, n_heads, seq_len, head_dim)

        # 6. 拼回维度并做输出投影
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)
