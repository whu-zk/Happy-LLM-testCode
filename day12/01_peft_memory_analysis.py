"""
第一阶段：打破显存焦虑
==================
通过数学计算理解为什么全参数微调这么"贵"，以及 PEFT 的解法
"""

import torch

# ============================================
# 第一部分：显存去哪了？
# ============================================

print("=" * 70)
print("【1】显存占用公式详解")
print("=" * 70)

print("""
训练显存公式:
┌─────────────────────────────────────────────────────────────────────┐
│  总显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值                      │
└─────────────────────────────────────────────────────────────────────┘

各组成部分:
┌─────────────────────────────────────────────────────────────────────┐
│  1. 模型参数 (Model Parameters)                                      │
│     - 存储权重矩阵                                                   │
│     - FP32: 4 bytes/param                                            │
│     - FP16/BF16: 2 bytes/param                                       │
│     - INT8: 1 byte/param                                             │
│                                                                     │
│  2. 梯度 (Gradients)                                                 │
│     - 反向传播计算的梯度                                             │
│     - 每个可训练参数对应一个梯度                                     │
│     - FP32: 4 bytes/param                                            │
│                                                                     │
│  3. 优化器状态 (Optimizer States)                                    │
│     - AdamW: 存储一阶矩(m)和二阶矩(v)                                │
│     - 每个可训练参数对应 2 个状态                                    │
│     - FP32: 8 bytes/param (4 + 4)                                    │
│                                                                     │
│  4. 激活值 (Activations)                                             │
│     - 前向传播中间结果，用于反向传播                                 │
│     - 与 batch size、序列长度相关                                    │
│     - Gradient Checkpointing 可以大幅减少                            │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：7B 模型显存计算
# ============================================

print("\n" + "=" * 70)
print("【2】7B 模型显存计算对比")
print("=" * 70)

def calculate_memory(model_size_billion, batch_size=1, seq_length=2048, 
                     precision="fp16", optimizer="adamw", grad_checkpoint=False):
    """
    计算训练显存占用
    
    Args:
        model_size_billion: 模型参数量（十亿）
        batch_size: 批大小
        seq_length: 序列长度
        precision: 精度 (fp32/fp16/bf16)
        optimizer: 优化器类型
        grad_checkpoint: 是否使用梯度检查点
    """
    params = model_size_billion * 1e9
    
    # 1. 模型参数显存
    if precision in ["fp16", "bf16"]:
        model_memory = params * 2  # 2 bytes
    else:  # fp32
        model_memory = params * 4  # 4 bytes
    
    # 2. 梯度显存 (训练时需要)
    gradient_memory = params * 4  # 梯度通常用 FP32
    
    # 3. 优化器状态显存
    if optimizer == "adamw":
        # AdamW: m + v, 每个 4 bytes
        optimizer_memory = params * 8
    elif optimizer == "sgd":
        optimizer_memory = params * 4
    else:
        optimizer_memory = params * 8
    
    # 4. 激活值显存 (简化估算)
    # 假设每层激活值 = batch_size * seq_length * hidden_size * 4 bytes
    # 7B 模型通常 32 层, hidden_size=4096
    layers = 32
    hidden_size = 4096
    activation_per_layer = batch_size * seq_length * hidden_size * 4
    
    if grad_checkpoint:
        # 梯度检查点只保存部分激活值
        activation_memory = activation_per_layer * 2  # 只保存输入
    else:
        activation_memory = activation_per_layer * layers
    
    # 总显存 (GB)
    total_memory = (model_memory + gradient_memory + optimizer_memory + activation_memory) / (1024**3)
    
    return {
        "model_params_gb": model_memory / (1024**3),
        "gradients_gb": gradient_memory / (1024**3),
        "optimizer_gb": optimizer_memory / (1024**3),
        "activations_gb": activation_memory / (1024**3),
        "total_gb": total_memory
    }

# 计算 7B 模型不同场景的显存占用
print("\n7B 模型显存占用计算:")
print("-" * 70)

scenarios = [
    ("推理 (FP16)", {"precision": "fp16", "optimizer": None, "grad_checkpoint": False}),
    ("推理 (INT8)", {"precision": "int8", "optimizer": None, "grad_checkpoint": False}),
    ("全参数训练 (FP16)", {"precision": "fp16", "optimizer": "adamw", "grad_checkpoint": False}),
    ("全参数训练 (FP16 + GC)", {"precision": "fp16", "optimizer": "adamw", "grad_checkpoint": True}),
]

for name, config in scenarios:
    if config["optimizer"] is None:
        # 推理模式
        params = 7e9
        if config["precision"] == "fp16":
            mem = params * 2 / (1024**3)
        elif config["precision"] == "int8":
            mem = params * 1 / (1024**3)
        print(f"\n{name}:")
        print(f"  显存占用: {mem:.1f} GB")
    else:
        # 训练模式
        mem = calculate_memory(
            7, 
            precision=config["precision"],
            optimizer=config["optimizer"],
            grad_checkpoint=config["grad_checkpoint"]
        )
        print(f"\n{name}:")
        print(f"  模型参数: {mem['model_params_gb']:.1f} GB")
        print(f"  梯度: {mem['gradients_gb']:.1f} GB")
        print(f"  优化器状态: {mem['optimizer_gb']:.1f} GB")
        print(f"  激活值: {mem['activations_gb']:.1f} GB")
        print(f"  总计: {mem['total_gb']:.1f} GB")

print("\n" + "=" * 70)
print("关键发现:")
print("=" * 70)
print("""
为什么 7B 模型推理只要 14GB，但训练需要近 160GB？

推理阶段:
  - 只需要加载模型参数
  - FP16: 7B × 2 bytes = 14 GB

训练阶段:
  - 模型参数: 14 GB (FP16)
  - 梯度: 28 GB (FP32)
  - 优化器状态: 56 GB (AdamW: 2×FP32)
  - 激活值: ~50+ GB (与 batch size 相关)
  - 总计: ~150 GB

差距: 150 / 14 ≈ 10 倍！
""")

# ============================================
# 第三部分：PEFT 家族概览
# ============================================

print("\n" + "=" * 70)
print("【3】PEFT (Parameter-Efficient Fine-Tuning) 家族")
print("=" * 70)

print("""
核心思想: 只训练少量参数，大幅降低显存需求

┌─────────────────────────────────────────────────────────────────────┐
│  1. Adapter Tuning (2019)                                            │
├─────────────────────────────────────────────────────────────────────┤
│  原理: 在 Transformer 层之间插入小型适配器模块                         │
│                                                                     │
│  结构: Input → [Adapter] → FFN → [Adapter] → Output                  │
│        其中 Adapter = Down-project → ReLU → Up-project               │
│                                                                     │
│  参数量: 原模型的 0.5% - 5%                                          │
│  优点: 简单有效，推理时可选择是否使用                                │
│  缺点: 增加推理延迟 (多了额外计算)                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  2. Prefix Tuning / Prompt Tuning (2021)                             │
├─────────────────────────────────────────────────────────────────────┤
│  原理: 在输入端添加可训练的"软提示"(Soft Prompt)                     │
│                                                                     │
│  结构: [Soft Prompt] + Input Tokens → Model                          │
│                                                                     │
│  Prefix Tuning: 在每一层都加前缀                                     │
│  Prompt Tuning: 只在输入层加前缀                                     │
│                                                                     │
│  参数量: 原模型的 0.01% - 0.1%                                       │
│  优点: 参数量极小，适合多任务切换                                    │
│  缺点: 效果通常不如 Adapter/LoRA                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  3. LoRA (Low-Rank Adaptation, 2021) ★ 当前绝对主流                 │
├─────────────────────────────────────────────────────────────────────┤
│  原理: 用低秩矩阵近似权重的更新                                      │
│                                                                     │
│  数学: W' = W + ΔW = W + BA                                          │
│        W: 预训练权重 (冻结)                                          │
│        B, A: 可训练的低秩矩阵 (r << d)                               │
│                                                                     │
│  参数量: 原模型的 0.1% - 1%                                          │
│  优点:                                                               │
│    - 不增加推理延迟 (可合并权重)                                     │
│    - 效果接近全参数微调                                              │
│    - 可切换不同任务的 LoRA 权重                                      │
│  缺点: 需要选择合适的 rank 值                                        │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第四部分：LoRA 显存节省计算
# ============================================

print("\n" + "=" * 70)
print("【4】LoRA 显存节省计算")
print("=" * 70)

def calculate_lora_memory(model_size_billion, lora_rank=8, target_modules_ratio=0.5):
    """
    计算 LoRA 微调显存占用
    
    Args:
        model_size_billion: 模型参数量（十亿）
        lora_rank: LoRA 秩
        target_modules_ratio: 应用 LoRA 的模块比例 (通常 attention 部分)
    """
    total_params = model_size_billion * 1e9
    
    # 假设只对部分模块应用 LoRA (如 attention)
    target_params = total_params * target_modules_ratio
    
    # LoRA 参数量 = 2 × rank × d × num_modules
    # 简化估算: 假设每个 target 参数对应 2 * rank 个参数
    lora_params = target_params * 2 * lora_rank / (4096)  # 假设 hidden_size=4096
    
    # 显存占用
    # 1. 模型参数 (FP16, 冻结)
    model_memory = total_params * 2 / (1024**3)
    
    # 2. LoRA 参数 (FP32, 可训练)
    lora_memory = lora_params * 4 / (1024**3)
    
    # 3. LoRA 梯度 (FP32)
    lora_gradient = lora_params * 4 / (1024**3)
    
    # 4. LoRA 优化器状态 (AdamW)
    lora_optimizer = lora_params * 8 / (1024**3)
    
    # 5. 激活值 (与 batch size 相关)
    activation_memory = 10  # 简化估算
    
    total = model_memory + lora_memory + lora_gradient + lora_optimizer + activation_memory
    
    return {
        "model_gb": model_memory,
        "lora_params_gb": lora_memory,
        "lora_gradients_gb": lora_gradient,
        "lora_optimizer_gb": lora_optimizer,
        "activations_gb": activation_memory,
        "total_gb": total,
        "lora_params_million": lora_params / 1e6,
        "trainable_ratio": lora_params / total_params * 100
    }

# 计算不同 LoRA 配置的显存
print("\n7B 模型 + LoRA 显存占用:")
print("-" * 70)

lora_configs = [
    ("LoRA (r=8)", {"lora_rank": 8}),
    ("LoRA (r=16)", {"lora_rank": 16}),
    ("LoRA (r=64)", {"lora_rank": 64}),
]

for name, config in lora_configs:
    mem = calculate_lora_memory(7, **config)
    print(f"\n{name}:")
    print(f"  LoRA 参数量: {mem['lora_params_million']:.1f}M ({mem['trainable_ratio']:.2f}%)")
    print(f"  模型参数 (冻结): {mem['model_gb']:.1f} GB")
    print(f"  LoRA 参数: {mem['lora_params_gb']:.2f} GB")
    print(f"  LoRA 梯度: {mem['lora_gradients_gb']:.2f} GB")
    print(f"  LoRA 优化器: {mem['lora_optimizer_gb']:.2f} GB")
    print(f"  激活值: {mem['activations_gb']:.1f} GB")
    print(f"  总计: {mem['total_gb']:.1f} GB")

print("\n" + "=" * 70)
print("对比总结:")
print("=" * 70)
print("""
7B 模型训练显存对比:
┌─────────────────────────┬─────────────┬────────────────┐
│  方法                   │  显存占用   │  可训练参数    │
├─────────────────────────┼─────────────┼────────────────┤
│  全参数训练 (FP16)      │  ~150 GB    │  100%          │
│  LoRA (r=8)             │  ~25 GB     │  0.1%          │
│  LoRA (r=16)            │  ~26 GB     │  0.2%          │
│  LoRA (r=64)            │  ~30 GB     │  0.8%          │
└─────────────────────────┴─────────────┴────────────────┘

节省效果:
  - 使用 LoRA (r=8): 显存从 150GB → 25GB，节省 83%！
  - 只需训练 0.1% 的参数，效果接近全参数微调
""")

# ============================================
# 第五部分：PEFT 库使用
# ============================================

print("\n" + "=" * 70)
print("【5】PEFT 库简介")
print("=" * 70)

print("""
Hugging Face PEFT 库:
┌─────────────────────────────────────────────────────────────────────┐
│  安装: pip install peft                                              │
│                                                                     │
│  支持的方法:                                                         │
│    - LoRA (推荐)                                                     │
│    - Prefix Tuning                                                   │
│    - Prompt Tuning                                                   │
│    - P-Tuning                                                        │
│    - AdaLoRA                                                         │
│    - IA³                                                             │
└─────────────────────────────────────────────────────────────────────┘

LoRA 使用示例:
┌─────────────────────────────────────────────────────────────────────┐
│  from peft import LoraConfig, get_peft_model, TaskType               │
│                                                                     │
│  # 配置 LoRA                                                         │
│  lora_config = LoraConfig(                                           │
│      r=16,                    # LoRA 秩                              │
│      lora_alpha=32,           # 缩放参数                             │
│      target_modules=["q_proj", "v_proj"],  # 目标模块                │
│      lora_dropout=0.05,       # Dropout                              │
│      bias="none",                                                    │
│      task_type=TaskType.CAUSAL_LM,                                   │
│  )                                                                   │
│                                                                     │
│  # 应用 LoRA                                                         │
│  model = get_peft_model(model, lora_config)                          │
│  model.print_trainable_parameters()  # 查看可训练参数                │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第六部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【6】关键要点总结")
print("=" * 70)

print("""
1. 显存占用公式
   总显存 ≈ 模型参数 + 梯度 + 优化器状态 + 激活值
   
2. 为什么训练比推理贵 10 倍？
   - 训练需要存储梯度、优化器状态、激活值
   - 推理只需要模型参数

3. PEFT 核心思想
   - 只训练少量参数，冻结大部分预训练权重
   - 大幅降低显存需求，保持模型效果

4. LoRA 优势
   - 当前绝对主流方法
   - 不增加推理延迟
   - 效果接近全参数微调
   - 可切换不同任务权重

5. 实践建议
   - 消费级 GPU (24GB): 使用 LoRA (r=8/16)
   - 专业级 GPU (40-80GB): 可考虑 LoRA (r=64) 或全参数
   - 多卡训练: 结合 DeepSpeed ZeRO 进一步节省显存
""")

print("\n" + "=" * 70)
print("第一阶段完成！")
print("=" * 70)
