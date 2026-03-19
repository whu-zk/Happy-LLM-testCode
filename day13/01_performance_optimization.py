"""
第一阶段：性能与显存极致优化
==============================
掌握显存优化的"三剑客"，在有限硬件下跑得更快
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：梯度检查点 (Gradient Checkpointing)
# ============================================

print("=" * 70)
print("【1】梯度检查点 (Gradient Checkpointing)")
print("=" * 70)

print("""
梯度检查点原理：
┌─────────────────────────────────────────────────────────────────────┐
│  问题：训练时需要保存激活值用于反向传播                              │
│        激活值占用大量显存！                                          │
│                                                                     │
│  解决方案：用"时间换空间"                                            │
│        - 前向传播：不保存中间激活值                                  │
│        - 反向传播：重新计算需要的激活值                              │
│                                                                     │
│  效果：                                                              │
│        - 显存节省：30%-50%                                           │
│        - 速度损失：约 20% (需要重新计算)                             │
└─────────────────────────────────────────────────────────────────────┘

显存占用对比：
┌─────────────────────┬─────────────────┬─────────────────┐
│  模型大小           │  无检查点       │  有检查点       │
├─────────────────────┼─────────────────┼─────────────────┤
│  7B 模型训练        │  ~80 GB         │  ~40-50 GB      │
│  13B 模型训练       │  ~150 GB        │  ~80-100 GB     │
└─────────────────────┴─────────────────┴─────────────────┘

代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  # 方式 1: 在模型上直接启用                                          │
│  model.gradient_checkpointing_enable()                               │
│                                                                     │
│  # 方式 2: 在 TrainingArguments 中启用                               │
│  training_args = TrainingArguments(                                  │
│      gradient_checkpointing=True,                                    │
│      ...                                                             │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：混合精度训练 FP16 vs BF16
# ============================================

print("\n" + "=" * 70)
print("【2】混合精度训练：FP16 vs BF16")
print("=" * 70)

print("""
为什么需要混合精度？
┌─────────────────────────────────────────────────────────────────────┐
│  FP32 (32位浮点):                                                  │
│    - 精度高，但显存占用大 (4 bytes/参数)                             │
│    - 7B 模型: 28 GB 显存                                           │
│                                                                     │
│  混合精度训练:                                                       │
│    - 前向/反向：使用 FP16/BF16 (2 bytes)                            │
│    - 优化器状态：保持 FP32 (保证精度)                                │
│    - 效果：速度更快，显存减半                                        │
└─────────────────────────────────────────────────────────────────────┘

FP16 vs BF16 对比：
┌─────────────────┬─────────────────────┬─────────────────────┐
│  特性           │  FP16               │  BF16               │
├─────────────────┼─────────────────────┼─────────────────────┤
│  指数位         │  5 bits             │  8 bits             │
│  尾数位         │  10 bits            │  7 bits             │
│  数值范围       │  较小 (容易溢出)    │  与 FP32 相同       │
│  精度           │  较高               │  略低 (但足够)      │
│  硬件支持       │  大部分 GPU         │  Ampere+ (30/40系)  │
│  训练稳定性     │  需要 loss scaling  │  更稳定             │
└─────────────────┴─────────────────────┴─────────────────────┘

为什么 Ampere (30/40系) 首选 BF16？
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ 数值范围与 FP32 相同，不会溢出                                   │
│  ✅ 不需要复杂的 loss scaling                                        │
│  ✅ 训练更稳定，不容易出现 NaN                                       │
│  ✅ 速度比 FP16 更快 (硬件优化)                                      │
└─────────────────────────────────────────────────────────────────────┘

代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  # 检查 GPU 是否支持 BF16                                            │
│  if torch.cuda.is_available() and torch.cuda.is_bf16_supported():    │
│      dtype = torch.bfloat16                                          │
│      use_bf16 = True                                                 │
│  else:                                                               │
│      dtype = torch.float16                                           │
│      use_bf16 = False                                                │
│                                                                     │
│  # 在 TrainingArguments 中设置                                       │
│  training_args = TrainingArguments(                                  │
│      bf16=use_bf16,          # 优先使用 BF16                         │
│      fp16=not use_bf16,      # 不支持则用 FP16                       │
│      ...                                                             │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# 检查 BF16 支持
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"\n当前 GPU: {gpu_name}")
    print(f"BF16 支持: {'✅ 支持' if bf16_supported else '❌ 不支持'}")
    
    if bf16_supported:
        print("  → 建议使用 BF16 训练")
    else:
        print("  → 使用 FP16 训练")
else:
    print("\n⚠️ 未检测到 GPU，使用 CPU 训练")

# ============================================
# 第三部分：Flash Attention 2.0
# ============================================

print("\n" + "=" * 70)
print("【3】Flash Attention 2.0")
print("=" * 70)

print("""
Flash Attention 解决的问题：
┌─────────────────────────────────────────────────────────────────────┐
│  标准 Attention 的瓶颈：                                             │
│                                                                     │
│  1. 显存占用高                                                       │
│     - Attention 矩阵: O(N²) 复杂度                                   │
│     - 序列长度 4096 → 16M 元素                                       │
│                                                                     │
│  2. 显存读写频繁                                                     │
│     - HBM (高带宽内存) 读写成为瓶颈                                  │
│     - 计算速度远快于显存访问                                         │
└─────────────────────────────────────────────────────────────────────┘

Flash Attention 原理：
┌─────────────────────────────────────────────────────────────────────┐
│  核心思想：分块计算 + 重计算                                         │
│                                                                     │
│  1. 分块 (Tiling)                                                   │
│     - 将大的 Attention 矩阵分成小块                                  │
│     - 每次只加载一小块到 SRAM (快速缓存)                             │
│                                                                     │
│  2. 在线 softmax                                                   │
│     - 不需要存储完整的 Attention 矩阵                                │
│     - 逐块计算 softmax，累积结果                                     │
│                                                                     │
│  3. 重计算 (Recomputation)                                          │
│     - 反向传播时不保存 Attention 矩阵                                │
│     - 需要时重新计算                                                 │
└─────────────────────────────────────────────────────────────────────┘

Flash Attention 效果：
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  序列长度       │  速度提升       │  显存节省       │  精度           │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│  512            │  2-3x           │  线性           │  相同           │
│  1024           │  3-5x           │  线性           │  相同           │
│  4096           │  5-8x           │  线性           │  相同           │
│  8192+          │  7-10x          │  线性           │  相同           │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

安装 Flash Attention：
┌─────────────────────────────────────────────────────────────────────┐
│  # 需要 CUDA 环境                                                    │
│  pip install flash-attn --no-build-isolation                         │
│                                                                     │
│  # 或使用预编译版本                                                  │
│  pip install flash-attn --find-links https://...                     │
└─────────────────────────────────────────────────────────────────────┘

代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  # 方式 1: 加载模型时指定                                            │
│  model = AutoModelForCausalLM.from_pretrained(                       │
│      model_name,                                                     │
│      attn_implementation="flash_attention_2",                        │
│      torch_dtype=torch.bfloat16,                                     │
│  )                                                                   │
│                                                                     │
│  # 方式 2: 使用 SDPA (PyTorch 2.0+ 默认)                             │
│  model = AutoModelForCausalLM.from_pretrained(                       │
│      model_name,                                                     │
│      attn_implementation="sdpa",                                     │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第四部分：实战整合三大优化技术
# ============================================

print("\n" + "=" * 70)
print("【4】实战：整合三大优化技术")
print("=" * 70)

# 检查环境
print("\n环境检查:")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  BF16 支持: {torch.cuda.is_bf16_supported()}")

# 检查 Flash Attention
try:
    import flash_attn
    flash_attn_available = True
    print(f"  Flash Attention: ✅ 已安装 (版本 {flash_attn.__version__})")
except ImportError:
    flash_attn_available = False
    print("  Flash Attention: ❌ 未安装")
    print("     安装命令: pip install flash-attn --no-build-isolation")

# 配置优化参数
print("\n" + "-" * 50)
print("优化配置:")
print("-" * 50)

# 确定最佳配置
if torch.cuda.is_available():
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    device_map = "auto"
else:
    use_bf16 = False
    use_fp16 = False
    device_map = None

use_flash_attn = flash_attn_available and torch.cuda.is_available()
attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

print(f"  混合精度: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
print(f"  Attention: {attn_impl}")
print(f"  梯度检查点: 启用")

# ============================================
# 第五部分：完整训练示例
# ============================================

print("\n" + "=" * 70)
print("【5】完整训练示例 (整合所有优化)")
print("=" * 70)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n加载模型: {model_name}")

# 加载模型 (使用 Flash Attention + BF16/FP16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation=attn_impl,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32,
    device_map=device_map,
)

print(f"✓ 模型加载完成")
print(f"  - Attention: {attn_impl}")
print(f"  - 数据类型: {model.dtype}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 启用梯度检查点
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

print("\n模型配置:")
print(f"  - 梯度检查点: ✅ 启用")
model.print_trainable_parameters()

# 加载数据
print("\n加载数据集...")
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:100]")

# 数据格式化
def format_data(example):
    instruction = example.get("instruction_zh", "") or example.get("instruction", "")
    input_text = example.get("input_zh", "") or example.get("input", "")
    output = example.get("output_zh", "") or example.get("output", "")
    
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

formatted_dataset = dataset.map(format_data)
formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col != "text"]
)

# 配置训练参数 (整合所有优化)
print("\n训练配置 (整合三大优化):")
print("-" * 50)

training_args = TrainingArguments(
    output_dir="./optimized_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=10,
    save_steps=100,
    
    # 优化 1: 混合精度
    bf16=use_bf16,
    fp16=use_fp16,
    
    # 优化 2: 梯度检查点
    gradient_checkpointing=True,
    
    # 其他优化
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)

print(f"  bf16: {training_args.bf16}")
print(f"  fp16: {training_args.fp16}")
print(f"  gradient_checkpointing: {training_args.gradient_checkpointing}")
print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

print("\n开始训练 (100 个样本)...")
print("=" * 50)

try:
    trainer.train()
    print("\n✓ 训练完成")
    
    # 保存模型
    model.save_pretrained("./optimized_output/adapter")
    tokenizer.save_pretrained("./optimized_output/adapter")
    print("✓ 模型已保存")
    
except KeyboardInterrupt:
    print("\n训练被中断")

# ============================================
# 第六部分：优化效果总结
# ============================================

print("\n" + "=" * 70)
print("【6】显存优化三剑客效果总结")
print("=" * 70)

print("""
优化技术对比 (7B 模型训练):
┌──────────────────────────┬─────────────┬─────────────┬─────────────┐
│  优化技术                │  显存节省   │  速度影响   │  实现难度   │
├──────────────────────────┼─────────────┼─────────────┼─────────────┤
│  梯度检查点              │  30-50%     │  -20%       │  ⭐ 简单    │
│  混合精度 (BF16/FP16)    │  40-50%     │  +30-50%    │  ⭐ 简单    │
│  Flash Attention 2       │  20-30%     │  +50-200%   │  ⭐⭐ 中等  │
├──────────────────────────┼─────────────┼─────────────┼─────────────┤
│  三者组合                │  60-80%     │  +50-100%   │  ⭐⭐ 中等  │
└──────────────────────────┴─────────────┴─────────────┴─────────────┘

显存占用估算 (7B 模型 + LoRA):
┌─────────────────────────────────┬─────────────────┐
│  配置                           │  显存需求       │
├─────────────────────────────────┼─────────────────┤
│  无优化 (FP32)                  │  ~150 GB        │
│  + 混合精度 (FP16)              │  ~80 GB         │
│  + 梯度检查点                   │  ~50 GB         │
│  + Flash Attention              │  ~40 GB         │
├─────────────────────────────────┼─────────────────┤
│  QLoRA (4-bit)                  │  ~8-12 GB       │
│  QLoRA + 梯度检查点             │  ~6-8 GB        │
└─────────────────────────────────┴─────────────────┘

推荐配置优先级:
┌─────────────────────────────────────────────────────────────────────┐
│  1. 必开: 混合精度 (BF16 > FP16)                                    │
│     → 几乎无缺点，速度更快显存更少                                  │
│                                                                     │
│  2. 推荐: 梯度检查点                                                │
│     → 显存紧张时开启，20% 速度换 50% 显存                           │
│                                                                     │
│  3. 进阶: Flash Attention 2                                         │
│     → 长序列 (2048+) 收益明显，需要安装                             │
│                                                                     │
│  4. 终极: QLoRA + 以上所有                                          │
│     → 消费级 GPU 也能训练 7B+ 模型                                  │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("第一阶段完成！")
print("=" * 70)
