import torch

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
