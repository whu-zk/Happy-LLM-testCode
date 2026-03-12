import torch
import torch.nn as nn
from day9_test1_RMSNorm import RMSNorm
from day9_test4_Attention import Attention
from day9_test5_TransformerBlock import TransformerBlock
from day9_test6_Llama import Llama

def _init_weights(module, n_layers=None):
    """
    大模型标准初始化策略：
    1. 线性层和 Embedding 使用 std=0.02 的正态分布
    2. 残差分支末尾的权重进行特殊缩放 (可选，这里采用标准初始化)
    """
    if isinstance(module, nn.Linear):
        # 遵循 GPT/Llama 的标准初始化
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f} M")
    return total
def smoke_test():
    # 1. 定义一个超小型的配置用于测试
    class ModelArgs:
        dim = 256
        n_layers = 4
        n_heads = 8
        n_kv_heads = 4 # 测试 GQA
        vocab_size = 32000
        multiple_of = 64
        norm_eps = 1e-5
        max_seq_len = 1024

    args = ModelArgs()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. 实例化模型
    print("正在构建 LLaMA 模型...")
    model = Llama(args)
    model.apply(lambda m: _init_weights(m, args.n_layers))
    
    # 3. 计算参数
    count_parameters(model)

    # 4. 构造随机输入: BatchSize=1, SeqLen=10
    tokens = torch.randint(0, args.vocab_size, (1, 10))
    
    # 5. 执行前向传播
    print("执行前向传播测试...")
    try:
        with torch.no_grad():
            output = model(tokens, start_pos=0)
        
        print(f"输入形状: {tokens.shape}")
        print(f"输出形状: {output.shape}")
        
        # 验证输出维度
        expected_shape = (1, 10, args.vocab_size)
        if output.shape == expected_shape:
            print("\n🎉 [SUCCESS]: LLaMA2 架构组装成功！维度完全匹配！")
        else:
            print(f"\n❌ [ERROR]: 维度不匹配。预期 {expected_shape}, 实际 {output.shape}")
            
    except Exception as e:
        print(f"\n❌ [CRASH]: 模型运行出错!")
        print(e)

smoke_test()
