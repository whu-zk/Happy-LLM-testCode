import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma (初始化为全 1)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 核心公式：x / sqrt(mean(x^2) + eps)
        # torch.rsqrt 是平方根的倒数，比 1/sqrt(x) 更快
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 关键工程细节：
        # 为了防止在计算平方和时溢出或精度不足，通常先转为 float32 计算，再转回原精度
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
