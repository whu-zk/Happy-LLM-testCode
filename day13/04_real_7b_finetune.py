"""
第四阶段：【模拟实战】微调一个真实 7B 模型
=============================================
在 24G 显存（如 3090/4090）环境下制定微调方案
"""

import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 第一部分：方案设计 - 24G 显存下的 7B 微调
# ============================================

print("=" * 70)
print("【1】方案设计：24G 显存环境下的 7B 模型微调")
print("=" * 70)

print("""
硬件环境：
┌─────────────────────────────────────────────────────────────────────┐
│  GPU: RTX 3090 / RTX 4090 (24GB VRAM)                               │
│  目标: 微调 7B-8B 级别大模型                                         │
└─────────────────────────────────────────────────────────────────────┘

模型选择：
┌──────────────────────────┬─────────────────┬─────────────────────────┐
│  模型                    │  大小           │  特点                   │
├──────────────────────────┼─────────────────┼─────────────────────────┤
│  meta-llama/Meta-Llama-3-8B   │  8B        │  英文强，开源可商用     │
│  Qwen/Qwen2-7B-Instruct  │  7B             │  中文强，阿里开源       │
│  01-ai/Yi-1.5-9B         │  9B             │  中文优化，零一万物     │
│  deepseek-ai/DeepSeek-7B │  7B             │  代码强，深度求索       │
└──────────────────────────┴─────────────────┴─────────────────────────┘

技术栈组合 (显存优化最大化)：
┌─────────────────────────────────────────────────────────────────────┐
│  1. 4-bit QLoRA (显存从 14GB → 4GB)                                 │
│     - load_in_4bit=True                                             │
│     - bnb_4bit_quant_type="nf4"                                     │
│     - bnb_4bit_use_double_quant=True                                │
│                                                                     │
│  2. BF16 混合精度 (速度 +30%)                                       │
│     - 3090/4090 支持 BF16                                           │
│     - bf16=True                                                     │
│                                                                     │
│  3. Flash Attention 2 (速度 +50-200%)                               │
│     - attn_implementation="flash_attention_2"                       │
│                                                                     │
│  4. Gradient Checkpointing (显存 -50%)                              │
│     - gradient_checkpointing=True                                   │
│                                                                     │
│  5. 分页优化器 (显存进一步优化)                                      │
│     - optim="paged_adamw_8bit"                                      │
└─────────────────────────────────────────────────────────────────────┘

显存预算 (7B 模型)：
┌─────────────────────────────────┬─────────────────┐
│  组件                           │  显存占用       │
├─────────────────────────────────┼─────────────────┤
│  4-bit 基座模型                 │  ~4 GB          │
│  LoRA 参数 (r=64)               │  ~100 MB        │
│  梯度                           │  ~200 MB        │
│  优化器状态 (8-bit)             │  ~800 MB        │
│  激活值 (checkpointing)         │  ~2-4 GB        │
├─────────────────────────────────┼─────────────────┤
│  总计                           │  ~8-10 GB       │
│  预留 (安全余量)                │  ~4-6 GB        │
├─────────────────────────────────┼─────────────────┤
│  建议最大 batch size            │  1-2            │
│  建议序列长度                   │  2048-4096      │
└─────────────────────────────────┴─────────────────┘
""")

# ============================================
# 第二部分：完整配置代码
# ============================================

print("\n" + "=" * 70)
print("【2】完整配置代码")
print("=" * 70)

print("""
>>> 完整配置代码：
""")

config_code = '''
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ========== 1. 配置 4-bit 量化 ==========
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 启用 4-bit
    bnb_4bit_quant_type="nf4",                  # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,      # BF16 计算
    bnb_4bit_use_double_quant=True,             # 双重量化
)

# ========== 2. 加载模型 (Flash Attention + BF16) ==========
model_name = "Qwen/Qwen2-7B-Instruct"  # 或 "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",    # Flash Attention 2
    device_map="auto",                          # 自动分配
    trust_remote_code=True,
)

# ========== 3. 准备模型用于训练 ==========
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# ========== 4. 配置 LoRA ==========
lora_config = LoraConfig(
    r=64,                    # 较大 rank，7B 模型可用
    lora_alpha=16,           # alpha = r/4 或 r/2
    target_modules=[         # 根据模型调整
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# ========== 5. 配置训练参数 ==========
training_args = SFTConfig(
    output_dir="./qwen2-7b-sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,      # 24G 显存建议 batch=1
    gradient_accumulation_steps=4,       # 等效 batch=4
    
    # 学习率配置
    learning_rate=2e-4,
    warmup_ratio=0.03,                   # 3% warmup
    lr_scheduler_type="cosine",
    
    # 稳定性配置
    max_grad_norm=0.3,                   # 保守梯度裁剪
    weight_decay=0.001,                  # 轻微正则化
    
    # 优化技术
    bf16=True,                           # BF16 混合精度
    gradient_checkpointing=True,         # 梯度检查点
    optim="paged_adamw_8bit",           # 分页优化器
    
    # 日志和评估
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # 其他
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="tensorboard",
)

# ========== 6. 初始化 Trainer ==========
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,           # 验证集用于监控过拟合
    args=training_args,
)

# ========== 7. 训练 ==========
trainer.train()
'''

print(config_code)

# ============================================
# 第三部分：实际可运行脚本
# ============================================

print("\n" + "=" * 70)
print("【3】实际可运行脚本 (使用小模型演示)")
print("=" * 70)

# 由于 7B 模型太大，使用 TinyLlama 演示完整配置流程
print("\n注意：使用 TinyLlama-1.1B 演示完整配置流程")
print("实际使用时只需将 model_name 改为 7B 模型\n")

# 检查环境
print("环境检查:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  BF16: {torch.cuda.is_bf16_supported()}")

# 检查 Flash Attention
try:
    import flash_attn
    flash_attn_available = True
    print(f"  Flash Attention: ✓ {flash_attn.__version__}")
except ImportError:
    flash_attn_available = False
    print("  Flash Attention: ✗ (pip install flash-attn)")

# 配置 4-bit 量化
print("\n" + "-" * 50)
print("配置 4-bit 量化...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("✓ BitsAndBytesConfig:")
print(f"  - load_in_4bit: True")
print(f"  - quant_type: nf4")
print(f"  - double_quant: True")

# 加载模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n加载模型: {model_name}")

attn_impl = "flash_attention_2" if flash_attn_available else "sdpa"

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 显存统计
    if hasattr(model, "get_memory_footprint"):
        mem_mb = model.get_memory_footprint() / 1024**2
        print(f"✓ 模型加载完成")
        print(f"  - 显存占用: {mem_mb:.1f} MB")
        print(f"  - Attention: {attn_impl}")
else:
    print("⚠️ 使用 CPU 模式加载")
    model = AutoModelForCausalLM.from_pretrained(model_name)

# 准备模型用于训练
print("\n准备模型用于训练...")
if torch.cuda.is_available():
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print("✓ prepare_model_for_kbit_training")

# 配置 LoRA (7B 模型可用更大 r)
print("\n配置 LoRA...")
lora_config = LoraConfig(
    r=64,                        # 7B 模型可用 r=64
    lora_alpha=16,               # alpha = r/4
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
print("✓ LoRA 配置完成")
print(f"  - r: {lora_config.r}")
print(f"  - alpha: {lora_config.lora_alpha}")
model.print_trainable_parameters()

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================
# 第四部分：数据准备
# ============================================

print("\n" + "=" * 70)
print("【4】数据准备")
print("=" * 70)

print("\n加载数据集...")

# 加载中文 Alpaca 数据
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")

# 分割训练集和验证集
dataset = dataset.shuffle(seed=42)
train_dataset = dataset.select(range(500))
eval_dataset = dataset.select(range(500, 600))

print(f"  训练集: {len(train_dataset)} 条")
print(f"  验证集: {len(eval_dataset)} 条")

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

print("\n格式化数据...")
train_dataset = train_dataset.map(format_data)
eval_dataset = eval_dataset.map(format_data)

train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c != "text"])
eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c != "text"])

print("✓ 数据准备完成")

# ============================================
# 第五部分：完整训练配置
# ============================================

print("\n" + "=" * 70)
print("【5】完整训练配置")
print("=" * 70)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

sft_config = SFTConfig(
    output_dir="./7b_sft_demo_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # 学习率
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # 稳定性
    max_grad_norm=0.3,
    weight_decay=0.001,
    
    # 优化技术
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
    
    # 评估 (关键！用于检测过拟合)
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # 日志
    logging_steps=10,
    logging_first_step=True,
    report_to="none",
    
    # 数据
    max_seq_length=512,
    dataset_text_field="text",
)

print("\n训练参数:")
print(f"  Batch size: {sft_config.per_device_train_batch_size}")
print(f"  Gradient accumulation: {sft_config.gradient_accumulation_steps}")
print(f"  等效 batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"  Learning rate: {sft_config.learning_rate}")
print(f"  Warmup ratio: {sft_config.warmup_ratio}")
print(f"  Max grad norm: {sft_config.max_grad_norm}")
print(f"  Weight decay: {sft_config.weight_decay}")
print(f"  BF16: {sft_config.bf16}")
print(f"  Eval steps: {sft_config.eval_steps}")

# ============================================
# 第六部分：过拟合监控
# ============================================

print("\n" + "=" * 70)
print("【6】过拟合监控：train_loss vs eval_loss")
print("=" * 70)

print("""
过拟合判断方法：
┌─────────────────────────────────────────────────────────────────────┐
│  正常训练:                                                           │
│    train_loss ↓    eval_loss ↓                                      │
│    两者同步下降，差距保持相对稳定                                    │
│                                                                     │
│  开始过拟合:                                                         │
│    train_loss ↓↓   eval_loss → 或 ↑                                 │
│    训练 Loss 继续降，验证 Loss 停滞或上升                            │
│                                                                     │
│  严重过拟合:                                                         │
│    train_loss ↓↓↓  eval_loss ↑↑                                     │
│    模型完全记住训练数据，失去泛化能力                                │
└─────────────────────────────────────────────────────────────────────┘

Loss 曲线可视化：
┌─────────────────────────────────────────────────────────────────────┐
│  正常:                      过拟合:                                 │
│                                                                     │
│  Loss  │  ╲ train              Loss  │  ╲ train                    │
│        │   ╲                    │    │   ╲  ╱ eval                │
│        │    ╲ eval              │    │    ╲╱                      │
│        │     ╲                  │    │     ╲                      │
│        └──────→                 │    └──────→                     │
│            Steps                     │        Steps                │
│                                                                     │
│  两线靠近同步下降                训练↓ 验证↑，差距拉大              │
└─────────────────────────────────────────────────────────────────────┘

应对策略：
┌──────────────────────────┬──────────────────────────────────────────┐
│  现象                    │  解决方案                                │
├──────────────────────────┼──────────────────────────────────────────┤
│  eval_loss 上升          │  提前停止 (Early Stopping)               │
│                          │  增大 weight_decay                       │
│                          │  减少训练轮数                            │
├──────────────────────────┼──────────────────────────────────────────┤
│  gap 过大 (差距>0.5)     │  增加训练数据                            │
│                          │  减小 LoRA rank                          │
│                          │  使用 dropout                            │
├──────────────────────────┼──────────────────────────────────────────┤
│  两者都不下降            │  增大学习率                              │
│                          │  检查数据质量                            │
└──────────────────────────┴──────────────────────────────────────────┘

关键配置：
┌─────────────────────────────────────────────────────────────────────┐
│  eval_strategy="steps"      # 定期评估                              │
│  eval_steps=100               # 每 100 step 评估一次                  │
│  load_best_model_at_end=True  # 加载最佳模型                         │
│  metric_for_best_model="eval_loss"  # 以 eval_loss 为指标            │
└─────────────────────────────────────────────────────────────────────┘
""")

# 自定义回调监控 Loss
class LossMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            
            if "loss" in logs:
                train_loss = logs["loss"]
                self.train_losses.append(train_loss)
                self.steps.append(step)
                
                if "eval_loss" in logs:
                    eval_loss = logs["eval_loss"]
                    self.eval_losses.append(eval_loss)
                    gap = train_loss - eval_loss
                    
                    print(f"  Step {step:3d} | Train Loss: {train_loss:.4f} | "
                          f"Eval Loss: {eval_loss:.4f} | Gap: {gap:+.4f}")
                    
                    # 过拟合警告
                    if len(self.eval_losses) >= 2:
                        if eval_loss > self.eval_losses[-2] and train_loss < self.train_losses[-2]:
                            print(f"  ⚠️  警告: 可能开始过拟合！")
                else:
                    print(f"  Step {step:3d} | Train Loss: {train_loss:.4f}")

# ============================================
# 第七部分：启动训练
# ============================================

print("\n" + "=" * 70)
print("【7】启动训练")
print("=" * 70)

# 初始化 trainer
loss_monitor = LossMonitorCallback()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
    callbacks=[loss_monitor],
)

print("\n开始训练...")
print("(训练 500 条，验证 100 条，约 125 steps)")
print("=" * 50)

try:
    trainer.train()
    
    print("\n" + "=" * 50)
    print("✓ 训练完成！")
    
    # 保存模型
    print("\n保存模型...")
    model.save_pretrained("./7b_sft_demo_output/final_adapter")
    tokenizer.save_pretrained("./7b_sft_demo_output/final_adapter")
    
    # 保存训练信息
    train_info = {
        "model_name": model_name,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": sft_config.learning_rate,
        "num_train_samples": len(train_dataset),
        "num_eval_samples": len(eval_dataset),
    }
    
    with open("./7b_sft_demo_output/training_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)
    
    print("✓ 模型和信息已保存到 ./7b_sft_demo_output/")
    
except KeyboardInterrupt:
    print("\n训练被中断")

# ============================================
# 第八部分：7B 模型实战检查清单
# ============================================

print("\n" + "=" * 70)
print("【8】7B 模型微调检查清单")
print("=" * 70)

print("""
环境准备：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ GPU 显存 >= 24GB (3090/4090)                                    │
│  ✅ CUDA >= 11.8                                                    │
│  ✅ 安装 flash-attn: pip install flash-attn --no-build-isolation   │
│  ✅ 安装 bitsandbytes: pip install bitsandbytes                     │
└─────────────────────────────────────────────────────────────────────┘

模型配置：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ 4-bit 量化: load_in_4bit=True                                   │
│  ✅ Flash Attention: attn_implementation="flash_attention_2"        │
│  ✅ BF16: bf16=True (3090/4090 支持)                                │
│  ✅ 梯度检查点: gradient_checkpointing=True                         │
│  ✅ 分页优化器: optim="paged_adamw_8bit"                           │
└─────────────────────────────────────────────────────────────────────┘

LoRA 配置 (7B 模型)：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ r=64 (或 32-128 范围)                                           │
│  ✅ alpha=16 (r/4)                                                  │
│  ✅ target_modules 包含所有 attention 和 MLP 层                      │
│  ✅ lora_dropout=0.05                                               │
└─────────────────────────────────────────────────────────────────────┘

训练配置：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ batch_size=1, gradient_accumulation_steps=4                     │
│  ✅ learning_rate=2e-4                                              │
│  ✅ warmup_ratio=0.03                                               │
│  ✅ max_grad_norm=0.3                                               │
│  ✅ weight_decay=0.001                                              │
│  ✅ 验证集 eval_dataset (用于监控过拟合)                            │
│  ✅ load_best_model_at_end=True                                     │
└─────────────────────────────────────────────────────────────────────┘

监控指标：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ train_loss 和 eval_loss 同步下降                                │
│  ✅ gap (train - eval) < 0.5                                        │
│  ✅ eval_loss 不持续上升                                            │
│  ✅ 保存最佳模型 (load_best_model_at_end)                           │
└─────────────────────────────────────────────────────────────────────┘

显存估算：
┌─────────────────────────────────────────────────────────────────────┐
│  7B 模型 4-bit:           ~4 GB                                     │
│  LoRA (r=64):             ~0.1 GB                                   │
│  梯度 + 优化器 (8-bit):   ~1 GB                                     │
│  激活值:                  ~3-4 GB                                   │
│  预留:                    ~4-6 GB                                   │
│  ─────────────────────────────────                                  │
│  总计:                    ~12-15 GB (24GB 显卡安全)                 │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("第四阶段完成！")
print("=" * 70)
