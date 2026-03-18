"""
第三阶段：PEFT 库实战编码
=======================
手写代码，将 LLaMA2 包装成一个 LoRA 模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ============================================
# 第一部分：LoraConfig 配置详解
# ============================================

print("=" * 70)
print("【1】LoraConfig 配置详解")
print("=" * 70)

print("""
LoraConfig 核心参数:
┌─────────────────────────────────────────────────────────────────────┐
│  r (int): LoRA 的秩                                                  │
│    - 控制低秩矩阵的维度                                              │
│    - 常用值: 8, 16, 32, 64                                           │
│    - 越大表达能力越强，参数量也越多                                  │
│                                                                     │
│  lora_alpha (int): 缩放参数                                          │
│    - 控制 LoRA 更新的强度                                            │
│    - 通常设为 r 的 2 倍                                              │
│    - 实际缩放比例 = lora_alpha / r                                   │
│                                                                     │
│  target_modules (List[str]): 目标模块                                │
│    - 指定对哪些层应用 LoRA                                           │
│    - LLaMA 常用: ["q_proj", "v_proj"] 或全部 attention               │
│                                                                     │
│  lora_dropout (float): Dropout 率                                    │
│    - 防止过拟合                                                      │
│    - 常用值: 0.0 - 0.1                                               │
│                                                                     │
│  bias (str): 偏置训练方式                                            │
│    - "none": 不训练偏置 (推荐)                                       │
│    - "all": 训练所有偏置                                             │
│    - "lora_only": 只训练 LoRA 层的偏置                               │
│                                                                     │
│  task_type (TaskType): 任务类型                                      │
│    - CAUSAL_LM: 因果语言模型 (如 GPT, LLaMA)                         │
│    - SEQ_CLS: 序列分类                                               │
│    - SEQ_2_SEQ_LM: 序列到序列 (如 T5)                                │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：实战代码 - 配置 LoraConfig
# ============================================

print("\n" + "=" * 70)
print("【2】实战：配置 LoraConfig")
print("=" * 70)

# 代码实现：配置 LoraConfig
print("\n>>> 代码实现：")
print("-" * 50)
print("""
from peft import LoraConfig, TaskType

# 配置 LoRA
config = LoraConfig(
    r=8,                          # LoRA 秩
    lora_alpha=32,                # 缩放参数 (alpha/r = 4)
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.05,            # Dropout
    bias="none",                  # 不训练偏置
    task_type=TaskType.CAUSAL_LM  # 因果语言模型
)
""")
print("-" * 50)

# 实际配置
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

print("\n✓ LoraConfig 配置完成")
print(f"  - r: {config.r}")
print(f"  - lora_alpha: {config.lora_alpha}")
print(f"  - 缩放比例: {config.lora_alpha / config.r}")
print(f"  - target_modules: {config.target_modules}")
print(f"  - lora_dropout: {config.lora_dropout}")
print(f"  - bias: {config.bias}")
print(f"  - task_type: {config.task_type}")

# ============================================
# 第三部分：加载模型与包装
# ============================================

print("\n" + "=" * 70)
print("【3】加载模型与包装")
print("=" * 70)

# 加载基座模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\n加载基座模型: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

total_params_before = sum(p.numel() for p in model.parameters())
print(f"✓ 基座模型加载完成")
print(f"  - 总参数量: {total_params_before:,} ({total_params_before/1e9:.2f}B)")

# 使用 get_peft_model 包装
print("\n>>> 代码实现：")
print("-" * 50)
print("""
from peft import get_peft_model

# 将 LoRA 配置应用到模型
model = get_peft_model(model, config)
""")
print("-" * 50)

# 实际包装
model = get_peft_model(model, config)
print("\n✓ LoRA 包装完成")

# ============================================
# 第四部分：打印可训练参数
# ============================================

print("\n" + "=" * 70)
print("【4】打印可训练参数")
print("=" * 70)

print("\n>>> 代码实现：")
print("-" * 50)
print("""
# 打印可训练参数信息
model.print_trainable_parameters()
""")
print("-" * 50)

print("\n>>> 执行结果：")
print("=" * 50)
model.print_trainable_parameters()
print("=" * 50)

# 手动计算验证
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_ratio = trainable_params / total_params * 100

print("\n手动验证计算:")
print(f"  总参数量: {total_params:,}")
print(f"  可训练参数量: {trainable_params:,}")
print(f"  可训练参数占比: {trainable_ratio:.4f}%")

# 验证是否低于 0.1%
if trainable_ratio < 0.1:
    print(f"\n✅ 成功！可训练参数占比 {trainable_ratio:.4f}% < 0.1%")
else:
    print(f"\n⚠️ 可训练参数占比 {trainable_ratio:.4f}%，略高于 0.1%")
    print("   可以尝试减小 r 值或 target_modules 数量")

# ============================================
# 第五部分：LoRA 模型结构分析
# ============================================

print("\n" + "=" * 70)
print("【5】LoRA 模型结构分析")
print("=" * 70)

print("\n模型中的 LoRA 层:")
print("-" * 50)

lora_layers = []
for name, module in model.named_modules():
    if "lora" in name.lower():
        lora_layers.append((name, module))

# 只显示前 10 个
for i, (name, module) in enumerate(lora_layers[:10]):
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params:,} params")

if len(lora_layers) > 10:
    print(f"  ... 还有 {len(lora_layers) - 10} 个 LoRA 层")

print(f"\n总计 LoRA 层数: {len(lora_layers)}")

# ============================================
# 第六部分：显存节省计算
# ============================================

print("\n" + "=" * 70)
print("【6】显存节省计算")
print("=" * 70)

# 计算显存占用
model_memory = total_params * 2 / (1024**3)  # FP16
lora_memory = trainable_params * 4 / (1024**3)  # FP32
lora_gradient = trainable_params * 4 / (1024**3)
lora_optimizer = trainable_params * 8 / (1024**3)
activation_memory = 5  # 估算

full_finetune_memory = total_params * 20 / (1024**3)  # 全参数训练估算
lora_total_memory = model_memory + lora_memory + lora_gradient + lora_optimizer + activation_memory

print("\n显存占用对比:")
print("-" * 50)
print(f"全参数微调估算:")
print(f"  模型参数: {model_memory:.1f} GB")
print(f"  梯度: {total_params * 4 / (1024**3):.1f} GB")
print(f"  优化器状态: {total_params * 8 / (1024**3):.1f} GB")
print(f"  激活值: ~10 GB")
print(f"  总计: ~{full_finetune_memory:.1f} GB")
print()
print(f"LoRA 微调:")
print(f"  模型参数 (冻结): {model_memory:.1f} GB")
print(f"  LoRA 参数: {lora_memory:.2f} GB")
print(f"  LoRA 梯度: {lora_gradient:.2f} GB")
print(f"  LoRA 优化器: {lora_optimizer:.2f} GB")
print(f"  激活值: ~{activation_memory} GB")
print(f"  总计: ~{lora_total_memory:.1f} GB")
print()
print(f"显存节省: {(1 - lora_total_memory/full_finetune_memory)*100:.1f}%")

# ============================================
# 第七部分：准备训练
# ============================================

print("\n" + "=" * 70)
print("【7】准备 LoRA 训练")
print("=" * 70)

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 加载数据
print("\n加载数据集...")
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:50]")
print(f"  样本数: {len(dataset)}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# 修复 gradient checkpointing 兼容性
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# 配置训练
sft_config = SFTConfig(
    output_dir="./lora_practice_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=5,
    logging_steps=5,
    save_steps=50,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("\n训练配置:")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(f"  Batch size: {sft_config.per_device_train_batch_size}")
print(f"  Learning rate: {sft_config.learning_rate}")

# ============================================
# 第八部分：启动训练
# ============================================

print("\n" + "=" * 70)
print("【8】启动 LoRA 训练")
print("=" * 70)

print("\n>>> 代码实现：")
print("-" * 50)
print("""
from trl import SFTTrainer

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
)

# 开始训练
trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./lora_adapter")
""")
print("-" * 50)

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
)

print("\n开始训练...")
print("(训练 50 个样本，约 13 个 step)\n")

try:
    trainer.train()
    
    # 保存
    print("\n保存 LoRA 权重...")
    model.save_pretrained("./lora_practice_output/adapter")
    tokenizer.save_pretrained("./lora_practice_output/adapter")
    print("✓ 训练完成，权重已保存")
    
except KeyboardInterrupt:
    print("\n训练被中断")

# ============================================
# 第九部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【9】PEFT 实战关键要点总结")
print("=" * 70)

print("""
1. LoRA 配置三要素
   - r: 控制表达能力 (8-64)
   - alpha: 控制更新强度 (通常 2*r)
   - target_modules: 决定应用范围

2. 包装流程
   from peft import LoraConfig, get_peft_model
   
   config = LoraConfig(...)
   model = get_peft_model(base_model, config)
   model.print_trainable_parameters()

3. 可训练参数验证
   - 目标: < 0.1% (本例: 0.06%)
   - 如果过高: 减小 r 或减少 target_modules

4. 训练技巧
   - LoRA 可以用更大学习率 (2e-4 vs 2e-5)
   - 记得 enable_input_require_grads()
   - 只保存 LoRA 权重，不保存完整模型

5. 部署方式
   - 动态加载: PeftModel.from_pretrained(base, lora_path)
   - 合并部署: model.merge_and_unload() 后保存
""")

print("\n" + "=" * 70)
print("第三阶段完成！")
print("=" * 70)
