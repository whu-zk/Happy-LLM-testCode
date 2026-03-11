import torch

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
