import torch.nn as nn
import torch
from day9_test1_RMSNorm import RMSNorm
from day9_test4_Attention import Attention
from day9_test2_FeedForward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # 1. 注意力分支
        self.attention = Attention(args) # Phase 2 实现的类
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps) # Phase 1 实现的类
        
        # 2. 前馈网络分支
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of
        ) # Phase 1 实现的类
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        # 第一部分：自注意力 + 残差连接
        # 注意：RMSNorm 在 Attention 之前 (Pre-Norm)
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        
        # 第二部分：前馈网络 + 残差连接
        # 注意：RMSNorm 在 FFN 之前 (Pre-Norm)
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out
