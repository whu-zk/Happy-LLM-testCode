import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v 的形状均为: (batch_size, num_heads, seq_len, d_k)
    """
    d_k = q.size(-1)
    
    # 1. 计算点积得分: Q * K^T
    # transpose(-2, -1) 是为了将最后两个维度转置，以便进行矩阵乘法
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 如果有掩码 (Mask)，将对应位置设为极小值，这样 Softmax 后的权重接近 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. Softmax 归一化得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和得到最终输出
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights
