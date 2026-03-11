import torch

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
