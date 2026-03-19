"""
第二阶段：稳定性与参数陷阱
==========================
通过调节超参数，让 Loss 曲线稳如老狗
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 第一部分：学习率 (LR) 与热身 (Warmup)
# ============================================

print("=" * 70)
print("【1】学习率 (LR) 与热身 (Warmup)")
print("=" * 70)

print("""
学习率的重要性：
┌─────────────────────────────────────────────────────────────────────┐
│  学习率太大 → 模型震荡，Loss 不收敛，甚至发散                        │
│  学习率太小 → 收敛极慢，可能陷入局部最优                             │
│  学习率合适 → 稳定下降，最终收敛到较好的解                           │
└─────────────────────────────────────────────────────────────────────┘

微调 vs 预训练的学习率：
┌─────────────────────────────────────────────────────────────────────┐
│  预训练学习率: 1e-4 到 1e-3                                          │
│  微调学习率:   1e-5 到 1e-4  (通常小 10 倍)                          │
│                                                                     │
│  原因:                                                               │
│    - 预训练: 模型从零学习，需要较大步伐                              │
│    - 微调: 模型已有知识，只需微调，步伐要小                          │
│                                                                     │
│  避坑指南:                                                           │
│    ⚠️ 如果模型输出乱码 → 首选尝试调小 LR                             │
│    ⚠️ 如果 Loss 不下降 → 尝试调大 LR                                 │
│    ⚠️ 如果 Loss 震荡 → 尝试减小 LR 或增大 batch size                 │
└─────────────────────────────────────────────────────────────────────┘

推荐学习率设置：
┌──────────────────────────┬─────────────────┬─────────────────────────┐
│  训练类型                │  推荐 LR        │  说明                   │
├──────────────────────────┼─────────────────┼─────────────────────────┤
│  全参数微调 (7B)         │  1e-5 ~ 2e-5    │  较小，防止破坏预训练   │
│  LoRA 微调               │  1e-4 ~ 2e-4    │  可稍大，参数量少       │
│  QLoRA 微调              │  2e-4 ~ 5e-4    │  可更大，4-bit 需要     │
│  预训练                  │  1e-4 ~ 1e-3    │  较大，从零学习         │
└──────────────────────────┴─────────────────┴─────────────────────────┘

Warmup (热身) 策略：
┌─────────────────────────────────────────────────────────────────────┐
│  问题：训练初期，模型参数随机初始化，梯度可能不稳定                    │
│        直接使用大学习率可能导致震荡                                  │
│                                                                     │
│  解决方案：Warmup                                                   │
│    - 前 N 个 step，学习率从 0 线性增加到目标值                       │
│    - 给模型"热身"时间，稳定后再全力训练                              │
│                                                                     │
│  Warmup 比例：                                                       │
│    - 通常占总 step 的 5%-10%                                         │
│    - 如 1000 个 step，warmup=100                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# 可视化学习率调度
def plot_lr_schedule(total_steps=1000, warmup_steps=100, base_lr=2e-4):
    """可视化不同学习率调度策略"""
    steps = np.arange(total_steps)
    
    # Linear warmup + cosine decay
    lr_cosine = []
    for step in steps:
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr_cosine.append(lr)
    
    # Linear warmup + linear decay
    lr_linear = []
    for step in steps:
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = base_lr * (1 - progress)
        lr_linear.append(lr)
    
    # Constant with warmup
    lr_constant = []
    for step in steps:
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            lr = base_lr
        lr_constant.append(lr)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, lr_cosine, label='Cosine Decay', linewidth=2)
    plt.plot(steps, lr_linear, label='Linear Decay', linewidth=2)
    plt.plot(steps, lr_constant, label='Constant', linewidth=2)
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 放大 warmup 阶段
    plt.plot(steps[:warmup_steps*2], lr_cosine[:warmup_steps*2], label='Cosine', linewidth=2)
    plt.axvline(x=warmup_steps, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Phase (Zoomed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=150)
    print("\n✓ 学习率调度图已保存: lr_schedule.png")
    plt.close()

plot_lr_schedule()

print("""
代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  from transformers import TrainingArguments                          │
│                                                                     │
│  training_args = TrainingArguments(                                  │
│      learning_rate=2e-4,              # 基础学习率                   │
│      warmup_steps=100,                # 热身步数                     │
│      lr_scheduler_type="cosine",      # 余弦衰减                     │
│      # 其他选项: "linear", "constant", "polynomial"                  │
│      ...                                                             │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：梯度裁剪 (Gradient Clipping)
# ============================================

print("\n" + "=" * 70)
print("【2】梯度裁剪 (Gradient Clipping)")
print("=" * 70)

print("""
梯度爆炸问题：
┌─────────────────────────────────────────────────────────────────────┐
│  现象：                                                              │
│    - Loss 突然变成 NaN 或无穷大                                      │
│    - Loss 曲线出现剧烈跳变 (Spike)                                   │
│    - 模型参数变成异常值                                              │
│                                                                     │
│  原因：                                                              │
│    - 某些样本梯度特别大                                              │
│    - 学习率过大                                                      │
│    - 模型深层梯度累积                                                │
└─────────────────────────────────────────────────────────────────────┘

梯度裁剪原理：
┌─────────────────────────────────────────────────────────────────────┐
│  计算梯度模长: ||g|| = sqrt(sum(g_i^2))                             │
│                                                                     │
│  如果 ||g|| > max_norm:                                             │
│      g_new = g * (max_norm / ||g||)                                 │
│                                                                     │
│  效果：强制梯度模长不超过阈值，防止爆炸                              │
└─────────────────────────────────────────────────────────────────────┘

梯度裁剪效果可视化：
┌─────────────────────────────────────────────────────────────────────┐
│  无裁剪:                    有裁剪 (max_norm=1.0):                  │
│                                                                     │
│  Loss    ↑  ╱╲                    Loss    ↑                         │
│          │ ╱  ╲   ╱╲                      │    ╱╲    ╱╲            │
│          │╱    ╲_╱  ╲____                 │___╱  ╲__╱  ╲___        │
│          └──────────────→                 └──────────────→          │
│              Steps                            Steps                 │
│                                                                     │
│  特点: 有剧烈 spike                特点: 平滑稳定                   │
└─────────────────────────────────────────────────────────────────────┘

推荐设置：
┌──────────────────────────┬─────────────────┬─────────────────────────┐
│  模型大小                │  max_norm       │  说明                   │
├──────────────────────────┼─────────────────┼─────────────────────────┤
│  小型模型 (< 1B)         │  1.0            │  保守设置               │
│  中型模型 (1B-7B)        │  0.5 - 1.0      │  常用范围               │
│  大型模型 (> 7B)         │  0.3 - 0.5      │  更保守                 │
└──────────────────────────┴─────────────────┴─────────────────────────┘

代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  training_args = TrainingArguments(                                  │
│      max_grad_norm=1.0,               # 梯度裁剪阈值                 │
│      # 设为 0 则禁用梯度裁剪                                         │
│      ...                                                             │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第三部分：权重衰减 (Weight Decay)
# ============================================

print("\n" + "=" * 70)
print("【3】权重衰减 (Weight Decay)")
print("=" * 70)

print("""
什么是权重衰减？
┌─────────────────────────────────────────────────────────────────────┐
│  权重衰减 = L2 正则化                                                │
│                                                                     │
│  损失函数: Loss = DataLoss + λ * ||w||²                             │
│                                                                     │
│  效果：                                                              │
│    - 惩罚大的权重值                                                  │
│    - 防止模型过拟合训练数据                                          │
│    - 鼓励模型学习更简单的模式                                        │
└─────────────────────────────────────────────────────────────────────┘

权重衰减 vs 过拟合：
┌─────────────────────────────────────────────────────────────────────┐
│  无权重衰减:                                                         │
│    - 模型可能过度记忆训练数据                                        │
│    - 训练 Loss ↓↓↓，验证 Loss ↑                                      │
│    - 泛化能力差                                                      │
│                                                                     │
│  有权重衰减 (weight_decay=0.01):                                     │
│    - 限制权重增长                                                    │
│    - 训练 Loss 和验证 Loss 更接近                                    │
│    - 更好的泛化能力                                                  │
└─────────────────────────────────────────────────────────────────────┘

推荐设置：
┌──────────────────────────┬─────────────────┬─────────────────────────┐
│  训练类型                │  weight_decay   │  说明                   │
├──────────────────────────┼─────────────────┼─────────────────────────┤
│  全参数微调              │  0.01 - 0.1     │  防止过拟合             │
│  LoRA 微调               │  0.0 - 0.01     │  参数少，可减小或关闭   │
│  预训练                  │  0.1            │  标准设置               │
└──────────────────────────┴─────────────────┴─────────────────────────┘

注意：权重衰减通常不应用于 bias 和 LayerNorm 参数
┌─────────────────────────────────────────────────────────────────────┐
│  # AdamW 优化器会自动处理                                            │
│  # 或者手动设置:                                                     │
│  decay_params = [p for n, p in model.named_parameters()             │
│                   if p.requires_grad and "bias" not in n]            │
│  no_decay_params = [p for n, p in model.named_parameters()          │
│                      if p.requires_grad and "bias" in n]             │
└─────────────────────────────────────────────────────────────────────┘

代码实现：
┌─────────────────────────────────────────────────────────────────────┐
│  training_args = TrainingArguments(                                  │
│      weight_decay=0.01,               # 权重衰减系数                 │
│      ...                                                             │
│  )                                                                   │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第四部分：完整稳定训练配置
# ============================================

print("\n" + "=" * 70)
print("【4】实战：完整稳定训练配置")
print("=" * 70)

# 加载模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n加载模型: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
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
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

print("\n模型配置完成")
model.print_trainable_parameters()

# 加载数据
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:100]")

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

# 稳定训练配置
print("\n" + "-" * 50)
print("稳定训练超参数配置:")
print("-" * 50)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

sft_config = SFTConfig(
    output_dir="./stable_training_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    
    # 学习率配置
    learning_rate=2e-4,                   # LoRA 推荐: 2e-4
    warmup_steps=20,                      # 5% of total steps
    lr_scheduler_type="cosine",           # 余弦衰减
    
    # 稳定性配置
    max_grad_norm=1.0,                    # 梯度裁剪
    weight_decay=0.01,                    # 权重衰减
    
    # 混合精度
    bf16=use_bf16,
    fp16=not use_bf16,
    
    # 日志和保存
    logging_steps=10,
    save_steps=50,
    logging_first_step=True,
    report_to="none",
    
    # 其他
    gradient_checkpointing=True,
    dataset_text_field="text",
    max_seq_length=512,
    seed=42,                              # 保证可复现
)

print(f"""
学习率相关:
  - learning_rate: {sft_config.learning_rate}     (LoRA 推荐 2e-4)
  - warmup_steps: {sft_config.warmup_steps}       (总 step 的 5-10%)
  - lr_scheduler_type: {sft_config.lr_scheduler_type}  (余弦衰减，软着陆)

稳定性相关:
  - max_grad_norm: {sft_config.max_grad_norm}     (梯度裁剪，防止 spike)
  - weight_decay: {sft_config.weight_decay}      (正则化，防止过拟合)

其他:
  - num_train_epochs: {sft_config.num_train_epochs}
  - seed: {sft_config.seed}                     (保证可复现)
""")

# 自定义回调监控训练
class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            lr = logs.get("learning_rate", 0)
            loss = logs.get("loss", 0)
            epoch = logs.get("epoch", 0)
            print(f"  Step {state.global_step:3d} | Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.2e}")

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
    callbacks=[TrainingMonitorCallback()],
)

print("\n开始训练 (监控 Loss 曲线)...")
print("=" * 50)

try:
    trainer.train()
    print("\n✓ 训练完成")
    
    # 保存模型
    model.save_pretrained("./stable_training_output/adapter")
    tokenizer.save_pretrained("./stable_training_output/adapter")
    print("✓ 模型已保存")
    
except KeyboardInterrupt:
    print("\n训练被中断")

# ============================================
# 第五部分：超参数调优指南
# ============================================

print("\n" + "=" * 70)
print("【5】超参数调优指南")
print("=" * 70)

print("""
问题诊断与解决方案：
┌──────────────────────────┬─────────────────────────┬─────────────────┐
│  问题现象                │  可能原因               │  解决方案       │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Loss = NaN             │  学习率过大 / 梯度爆炸  │  降低 LR        │
│                          │                         │  降低 max_grad_norm │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Loss 震荡不下降         │  学习率过大             │  降低 LR        │
│                          │  batch size 太小        │  增大 batch     │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Loss 下降极慢           │  学习率过小             │  增大 LR        │
│                          │  warmup 过长            │  减少 warmup    │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  训练 Loss ↓ 验证 Loss ↑ │  过拟合                 │  增大 weight_decay │
│                          │                         │  减少训练轮数   │
├──────────────────────────┼─────────────────────────┼─────────────────┤
│  Loss 突然 spike         │  异常样本               │  启用梯度裁剪   │
│                          │  梯度累积不稳定         │  检查数据质量   │
└──────────────────────────┴─────────────────────────┴─────────────────┘

推荐调参顺序：
┌─────────────────────────────────────────────────────────────────────┐
│  1. 先确定学习率 (最重要)                                           │
│     - 从 2e-4 (LoRA) 或 1e-5 (全参数) 开始                          │
│     - 观察前 100 step 的 Loss 曲线                                  │
│     - 如果震荡 → 降低 10 倍                                         │
│     - 如果不动 → 提高 10 倍                                         │
│                                                                     │
│  2. 设置 warmup (5-10% of total steps)                              │
│                                                                     │
│  3. 启用梯度裁剪 (max_grad_norm=1.0)                                │
│     - 保险措施，防止意外 spike                                      │
│                                                                     │
│  4. 调整 weight_decay (LoRA 可设为 0)                               │
│     - 如果过拟合 → 增大                                             │
│     - 如果欠拟合 → 减小或关闭                                       │
│                                                                     │
│  5. 选择学习率调度器                                                │
│     - cosine: 平滑衰减，推荐                                        │
│     - linear: 线性衰减                                              │
│     - constant: 保持不变                                            │
└─────────────────────────────────────────────────────────────────────┘

稳定训练检查清单：
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ 学习率合适 (不震荡、不停滞)                                      │
│  ✅ 有 warmup 阶段                                                   │
│  ✅ 启用梯度裁剪                                                     │
│  ✅ 设置 weight_decay (适当正则化)                                   │
│  ✅ 使用 cosine 学习率衰减                                           │
│  ✅ 设置随机种子 (保证可复现)                                        │
│  ✅ 监控训练和验证 Loss                                              │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("第二阶段完成！")
print("=" * 70)
