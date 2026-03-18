"""
第四阶段：进阶黑科技——QLoRA
============================
学习如何在极低显存（单卡 8G-12G）下微调 7B 级别模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import os

# ============================================
# 第一部分：什么是 QLoRA？
# ============================================

print("=" * 70)
print("【1】QLoRA 原理详解")
print("=" * 70)

print("""
QLoRA (Quantized LoRA) 核心思想:
┌─────────────────────────────────────────────────────────────────────┐
│  4-bit 量化 + LoRA = 极低显存微调大模型                              │
└─────────────────────────────────────────────────────────────────────┘

显存占用对比 (7B 模型):
┌─────────────────────────┬─────────────┬────────────────┐
│  方法                   │  基座显存   │  训练总显存    │
├─────────────────────────┼─────────────┼────────────────┤
│  全参数训练 (FP16)      │  14 GB      │  ~150 GB       │
│  LoRA (FP16)            │  14 GB      │  ~25 GB        │
│  QLoRA (4-bit + LoRA)   │  ~4 GB      │  ~8-12 GB      │
└─────────────────────────┴─────────────┴────────────────┘

QLoRA 三大核心技术:
┌─────────────────────────────────────────────────────────────────────┐
│  1. 4-bit Normal Float (NF4) 量化                                    │
│     - 将权重从 FP16 (16-bit) 压缩到 4-bit                            │
│     - 使用分位数量化，保持模型精度                                   │
│     - 7B 模型从 14GB → 3.5GB                                         │
│                                                                     │
│  2. Double Quantization (双重量化)                                   │
│     - 对量化常数再次量化                                             │
│     - 进一步节省显存 (~0.5GB → ~0.1GB)                               │
│     - 几乎不损失精度                                                 │
│                                                                     │
│  3. Paged Optimizers (分页优化器)                                    │
│     - 使用 CPU 内存作为 GPU 显存的"分页"                             │
│     - 自动在 CPU-GPU 间交换优化器状态                                │
│     - 避免 OOM (Out of Memory)                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：BitsAndBytesConfig 配置
# ============================================

print("\n" + "=" * 70)
print("【2】BitsAndBytesConfig 配置详解")
print("=" * 70)

print("""
BitsAndBytesConfig 核心参数:
┌─────────────────────────────────────────────────────────────────────┐
│  load_in_4bit (bool): 启用 4-bit 量化                                │
│    - True: 使用 4-bit 量化                                           │
│    - False: 使用 8-bit 量化 (如果 load_in_8bit=True)                 │
│                                                                     │
│  bnb_4bit_compute_dtype (torch.dtype): 计算精度                      │
│    - torch.float16: 半精度计算                                       │
│    - torch.bfloat16: BF16 计算 (推荐，更稳定)                        │
│    - 计算时临时反量化到该精度                                        │
│                                                                     │
│  bnb_4bit_quant_type (str): 量化类型                                 │
│    - "nf4": Normal Float 4 (推荐，QLoRA 默认)                        │
│    - "fp4": 标准 FP4 量化                                            │
│                                                                     │
│  bnb_4bit_use_double_quant (bool): 双重量化                          │
│    - True: 启用双重量化，进一步节省显存                              │
│    - 对量化常数再次量化                                              │
│                                                                     │
│  bnb_4bit_quant_storage (torch.dtype): 存储类型                      │
│    - 量化权重的存储类型                                              │
│    - 通常设为 uint8                                                  │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第三部分：配置 QLoRA
# ============================================

print("\n" + "=" * 70)
print("【3】配置 QLoRA")
print("=" * 70)

print("\n>>> 代码实现：")
print("-" * 50)
print("""
from transformers import BitsAndBytesConfig

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 启用 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,      # 计算使用 BF16
    bnb_4bit_quant_type="nf4",                  # NF4 量化类型
    bnb_4bit_use_double_quant=True,             # 启用双重量化
)
""")
print("-" * 50)

# 实际配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("\n✓ BitsAndBytesConfig 配置完成")
print(f"  - load_in_4bit: {bnb_config.load_in_4bit}")
print(f"  - compute_dtype: {bnb_config.bnb_4bit_compute_dtype}")
print(f"  - quant_type: {bnb_config.bnb_4bit_quant_type}")
print(f"  - use_double_quant: {bnb_config.bnb_4bit_use_double_quant}")

# ============================================
# 第四部分：加载 4-bit 量化模型
# ============================================

print("\n" + "=" * 70)
print("【4】加载 4-bit 量化模型")
print("=" * 70)

# 使用 TinyLlama 作为演示 (QLoRA 可以支持更大的模型)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n加载模型: {model_name}")
print("(使用 4-bit 量化)")

print("\n>>> 代码实现：")
print("-" * 50)
print("""
from transformers import AutoModelForCausalLM

# 加载 4-bit 量化模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,      # 传入量化配置
    device_map="auto",                    # 自动分配层到设备
    trust_remote_code=True,
)
""")
print("-" * 50)

# 检查是否有 GPU
if torch.cuda.is_available():
    print("\n检测到 GPU，启用 4-bit 量化...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 计算显存占用
    if hasattr(model, "get_memory_footprint"):
        memory_mb = model.get_memory_footprint() / (1024 ** 2)
        print(f"\n✓ 4-bit 模型加载完成")
        print(f"  - 显存占用: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
    else:
        print(f"\n✓ 4-bit 模型加载完成")
        
    # 对比 FP16 显存
    fp16_memory = 1.1 * 2  # 1.1B params * 2 bytes
    print(f"  - FP16 预估占用: {fp16_memory:.2f} GB")
    print(f"  - 节省显存: {(1 - memory_mb/1024/fp16_memory)*100:.1f}%")
else:
    print("\n⚠️ 未检测到 GPU，使用 FP32 加载 (CPU 环境)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================
# 第五部分：准备模型用于训练
# ============================================

print("\n" + "=" * 70)
print("【5】准备模型用于训练")
print("=" * 70)

print("""
为什么需要 prepare_model_for_kbit_training?
┌─────────────────────────────────────────────────────────────────────┐
│  4-bit 量化模型的特殊处理：                                          │
│                                                                     │
│  1. 梯度检查点兼容                                                   │
│     - 量化模型需要特殊的梯度检查点设置                               │
│                                                                     │
│  2. 输入梯度启用                                                     │
│     - 确保输入张量可以计算梯度                                       │
│                                                                     │
│  3. 归一化层精度保持                                                 │
│     - LayerNorm 等层保持 FP32 精度                                   │
│     - 避免量化误差累积                                               │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n>>> 代码实现：")
print("-" * 50)
print("""
from peft import prepare_model_for_kbit_training

# 准备量化模型用于训练
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,    # 启用梯度检查点
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
""")
print("-" * 50)

# 实际准备
if torch.cuda.is_available():
    print("\n准备量化模型用于训练...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print("✓ 模型准备完成")

# ============================================
# 第六部分：配置 LoRA
# ============================================

print("\n" + "=" * 70)
print("【6】配置 LoRA")
print("=" * 70)

lora_config = LoraConfig(
    r=16,                           # LoRA 秩
    lora_alpha=32,                  # 缩放参数
    target_modules=[                # 目标模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,              # Dropout
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("LoRA 配置:")
print(f"  - r: {lora_config.r}")
print(f"  - lora_alpha: {lora_config.lora_alpha}")
print(f"  - target_modules: {lora_config.target_modules}")

# 应用 LoRA
print("\n应用 LoRA...")
model = get_peft_model(model, lora_config)

# 打印可训练参数
print("\n" + "=" * 50)
model.print_trainable_parameters()
print("=" * 50)

# ============================================
# 第七部分：双重量化与分页优化器详解
# ============================================

print("\n" + "=" * 70)
print("【7】双重量化与分页优化器详解")
print("=" * 70)

print("""
双重量化 (Double Quantization):
┌─────────────────────────────────────────────────────────────────────┐
│  问题：量化本身也需要存储常数 (scale, zero_point)                     │
│        每个量化块需要一个 32-bit 常数                                │
│                                                                     │
│  解决方案：对这些常数再次量化！                                      │
│                                                                     │
│  效果：                                                              │
│    - 每个块节省: 32-bit → 8-bit                                      │
│    - 对于 7B 模型: 节省 ~0.4GB 显存                                  │
│    - 精度损失: 可忽略不计                                            │
│                                                                     │
│  配置: bnb_4bit_use_double_quant=True                               │
└─────────────────────────────────────────────────────────────────────┘

分页优化器 (Paged Optimizers):
┌─────────────────────────────────────────────────────────────────────┐
│  问题：优化器状态占用大量显存 (AdamW: 2×模型大小)                    │
│        7B 模型优化器状态: ~56GB (FP32)                               │
│                                                                     │
│  解决方案：使用 CPU 内存作为"分页"                                   │
│                                                                     │
│  原理：                                                              │
│    - 不常用的优化器状态存到 CPU 内存                                 │
│    - 需要时自动交换到 GPU                                            │
│    - 类似操作系统的虚拟内存机制                                      │
│                                                                     │
│  配置: 使用 bitsandbytes 的 8-bit 优化器                             │
│    optim="paged_adamw_8bit"                                          │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第八部分：准备数据并训练
# ============================================

print("\n" + "=" * 70)
print("【8】准备数据并训练")
print("=" * 70)

# 加载数据
print("\n加载数据集...")
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:50]")
print(f"  样本数: {len(dataset)}")

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

# 配置训练
sft_config = SFTConfig(
    output_dir="./qlora_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,      # QLoRA 可以用更大的 batch
    gradient_accumulation_steps=4,       # 等效 batch=4
    learning_rate=2e-4,
    warmup_steps=5,
    logging_steps=5,
    save_steps=50,
    # 分页优化器 (节省显存)
    optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
    fp16=False,                         # 4-bit 模型通常用 bf16
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("\n训练配置:")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(f"  Batch size: {sft_config.per_device_train_batch_size}")
print(f"  Optimizer: {sft_config.optim}")
print(f"  Learning rate: {sft_config.learning_rate}")

# ============================================
# 第九部分：启动 QLoRA 训练
# ============================================

print("\n" + "=" * 70)
print("【9】启动 QLoRA 训练")
print("=" * 70)

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
)

print("\n开始训练...")
print("(QLoRA 训练 50 个样本)\n")

try:
    trainer.train()
    
    # 保存
    print("\n保存 QLoRA 权重...")
    model.save_pretrained("./qlora_output/adapter")
    tokenizer.save_pretrained("./qlora_output/adapter")
    print("✓ 训练完成，权重已保存")
    
except KeyboardInterrupt:
    print("\n训练被中断")

# ============================================
# 第十部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【10】QLoRA 关键要点总结")
print("=" * 70)

print("""
1. QLoRA 核心公式
   4-bit 量化 + LoRA = 极低显存微调
   7B 模型: 150GB → 8-12GB (节省 90%+)

2. 三大核心技术
   - NF4 量化: 16-bit → 4-bit
   - 双重量化: 量化常数再次量化
   - 分页优化器: CPU 内存作为 GPU 分页

3. 关键配置
   BitsAndBytesConfig:
     - load_in_4bit=True
     - bnb_4bit_quant_type="nf4"
     - bnb_4bit_use_double_quant=True
   
   TrainingArguments:
     - optim="paged_adamw_8bit"
     - prepare_model_for_kbit_training()

4. 适用场景
   - 消费级 GPU (8-12GB): 可以微调 7B 模型
   - 专业级 GPU (24GB): 可以微调 13B-30B 模型
   - 多卡训练: 可以微调 70B+ 模型

5. 注意事项
   - 4-bit 量化有轻微精度损失
   - 不适合对精度要求极高的任务
   - 推理时需要反量化，速度略慢
""")

print("\n" + "=" * 70)
print("第四阶段完成！")
print("=" * 70)
