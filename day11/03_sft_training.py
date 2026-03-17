"""
第三阶段：有监督微调 (SFT) 实战
==============================
使用 trl 库启动第一个微调任务
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import os

# ============================================
# 第一部分：配置训练参数详解
# ============================================

print("=" * 70)
print("【1】TrainingArguments 参数配置详解")
print("=" * 70)

print("""
核心参数说明:
┌─────────────────────────────────────────────────────────────────────┐
│  训练效率参数                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  per_device_train_batch_size: 每张卡的批大小                         │
│    - 显存充足: 可设为 4, 8, 16                                       │
│    - 显存不足: 设为 1 或 2                                           │
│                                                                     │
│  gradient_accumulation_steps: 梯度累加步数                           │
│    - 显存不足时的"救命稻草"                                          │
│    - 实际 batch_size = per_device_batch * accumulation * num_gpus   │
│    - 例如: batch=2, accumulation=4 → 等效 batch=8                    │
│                                                                     │
│  num_train_epochs: 训练轮数                                          │
│    - SFT 通常 1-3 轮即可                                             │
│    - 过多会导致过拟合                                                │
├─────────────────────────────────────────────────────────────────────┤
│  优化参数                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  learning_rate: 学习率                                               │
│    - 预训练: 1e-4 ~ 1e-3                                             │
│    - SFT 微调: 1e-5 ~ 5e-5 (推荐 2e-5)                               │
│    - 太小: 收敛慢；太大: 不稳定                                      │
│                                                                     │
│  warmup_steps: 预热步数                                              │
│    - 前几步学习率从 0 逐渐上升到设定值                               │
│    - 有助于训练稳定性                                                │
│                                                                     │
│  weight_decay: 权重衰减                                              │
│    - L2 正则化，防止过拟合                                           │
│    - 通常设为 0.01 或 0.001                                          │
├─────────────────────────────────────────────────────────────────────┤
│  显存优化参数                                                        │
├─────────────────────────────────────────────────────────────────────┤
│  fp16/bf16: 混合精度训练                                             │
│    - 减少显存占用，加速训练                                          │
│    - bf16 更稳定，但需 Ampere 以上 GPU                               │
│                                                                     │
│  gradient_checkpointing: 梯度检查点                                  │
│    - 时间换空间，大幅减少显存                                        │
│    - 训练速度会降低约 20%                                            │
│                                                                     │
│  optim: 优化器选择                                                   │
│    - adamw_torch: 标准 AdamW                                         │
│    - adamw_8bit: 8bit 优化器，省显存                                 │
│    - paged_adamw_8bit: 分页 8bit，超大模型适用                       │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：加载模型和数据
# ============================================

print("\n" + "=" * 70)
print("【2】加载模型和数据集")
print("=" * 70)

# 模型配置
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./sft_output"

print(f"\n加载模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型 (使用 fp16 节省显存)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print(f"模型加载完成")
print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"  - 设备: {next(model.parameters()).device}")

# 加载数据集 (使用中文 Alpaca 数据)
print("\n加载数据集: silk-road/alpaca-data-gpt4-chinese")
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:100]")
print(f"  - 样本数量: {len(dataset)}")

# ============================================
# 第三部分：数据预处理
# ============================================

print("\n" + "=" * 70)
print("【3】数据预处理与 Chat Template")
print("=" * 70)

def format_alpaca_to_chat(example):
    """将 Alpaca 格式转换为 Chat 格式并应用 Template"""
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
    
    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

# 格式化数据集
print("\n格式化数据集...")
formatted_dataset = dataset.map(format_alpaca_to_chat)

# 只保留 text 列，移除其他列
formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col != "text"]
)

# 查看示例
print("\n格式化后的示例:")
print("-" * 50)
print(formatted_dataset[0]["text"][:500])
print("-" * 50)

# ============================================
# 第四部分：配置 TrainingArguments 和 SFTConfig
# ============================================

print("\n" + "=" * 70)
print("【4】配置 TrainingArguments 和 SFTConfig")
print("=" * 70)

# 基础训练参数
training_args = TrainingArguments(
    # 输出目录
    output_dir=output_dir,
    
    # 训练轮数
    num_train_epochs=1,
    
    # 批大小配置
    per_device_train_batch_size=1,      # 单卡 batch size
    gradient_accumulation_steps=4,       # 梯度累加 4 步
    # 等效 batch size = 1 * 4 = 4
    
    # 学习率配置
    learning_rate=2e-5,                  # SFT 推荐学习率
    warmup_steps=10,                     # 预热步数
    weight_decay=0.01,                   # 权重衰减
    
    # 日志和保存
    logging_steps=5,                     # 每 5 步记录日志
    save_steps=50,                       # 每 50 步保存
    save_total_limit=2,                  # 最多保留 2 个 checkpoint
    
    # 优化器
    optim="adamw_torch",
    
    # 混合精度 (CPU 环境自动关闭)
    fp16=torch.cuda.is_available(),
    
    # 梯度检查点 (节省显存)
    gradient_checkpointing=True,
    
    # 其他
    remove_unused_columns=False,
    report_to="none",                    # 不使用 wandb/tensorboard
)

# SFT 专用配置
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    optim="adamw_torch",
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    # SFT 特有参数
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("训练参数配置:")
print(f"  - Epochs: {sft_config.num_train_epochs}")
print(f"  - Batch size per device: {sft_config.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {sft_config.gradient_accumulation_steps}")
print(f"  - Effective batch size: {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"  - Learning rate: {sft_config.learning_rate}")
print(f"  - Warmup steps: {sft_config.warmup_steps}")
print(f"  - FP16: {sft_config.fp16}")
print(f"  - Gradient checkpointing: {sft_config.gradient_checkpointing}")
print(f"  - Max sequence length: {sft_config.max_seq_length}")
print(f"  - Packing: {sft_config.packing}")

# ============================================
# 第五部分：SFTTrainer 详解
# ============================================

print("\n" + "=" * 70)
print("【5】SFTTrainer 配置详解")
print("=" * 70)

print("""
SFTTrainer 关键参数:
┌─────────────────────────────────────────────────────────────────────┐
│  dataset_text_field: 指定数据集中哪个字段包含文本                    │
│    - 必须设置为格式化后的文本字段名                                  │
│    - 本例中设为 "text"                                               │
│                                                                     │
│  max_seq_length: 最大序列长度                                        │
│    - 超过此长度的样本会被截断                                        │
│    - 根据显存调整，通常 512, 1024, 2048                              │
│                                                                     │
│  packing: 是否启用样本打包                                           │
│    - True: 将多个短样本拼接成一个长序列                              │
│    - 优点: 提高训练效率，减少 padding                                │
│    - 缺点: 可能在样本边界处产生干扰                                  │
│    - 建议: 短文本数据集启用，长文本禁用                              │
│                                                                     │
│  data_collator: 数据整理器                                           │
│    - 自动处理 padding 和 batch 组装                                  │
└─────────────────────────────────────────────────────────────────────┘
""")

# 初始化 SFTTrainer
print("\n初始化 SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
)

print("SFTTrainer 初始化完成!")

# ============================================
# 第六部分：启动训练
# ============================================

print("\n" + "=" * 70)
print("【6】启动训练")
print("=" * 70)

print("\n开始训练...")
print("(按 Ctrl+C 可中断训练)\n")

try:
    trainer.train()
    
    # 保存最终模型
    print("\n保存最终模型...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"✓ 模型已保存到: {output_dir}/final_model")
    
except KeyboardInterrupt:
    print("\n\n训练被用户中断")
    print("保存当前进度...")
    trainer.save_model(os.path.join(output_dir, "interrupted_model"))
    print(f"✓ 模型已保存到: {output_dir}/interrupted_model")

# ============================================
# 第七部分：训练后测试
# ============================================

print("\n" + "=" * 70)
print("【7】训练后测试")
print("=" * 70)

def generate_response(model, tokenizer, prompt, max_length=200):
    """生成回复"""
    messages = [{"role": "user", "content": prompt}]
    
    # 应用 chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 添加 assistant 标记
    )
    
    # 编码
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 的回复
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

# 测试问题
test_prompts = [
    "请介绍一下机器学习是什么",
    "给出三个保持健康的小贴士",
]

print("\n测试模型生成:")
for prompt in test_prompts:
    print(f"\n用户: {prompt}")
    response = generate_response(model, tokenizer, prompt)
    print(f"助手: {response[:200]}...")

# ============================================
# 第八部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【8】SFT 训练关键要点总结")
print("=" * 70)

print("""
1. 显存优化三板斧
   - gradient_accumulation_steps: 小 batch 模拟大 batch
   - gradient_checkpointing: 时间换空间
   - fp16/bf16: 混合精度训练

2. 学习率选择
   - SFT 使用较小学习率 (1e-5 ~ 5e-5)
   - 配合 warmup 稳定训练初期

3. 数据格式
   - 必须使用模型的 Chat Template
   - dataset_text_field 指向格式化后的文本

4. Packing 策略
   - 短文本数据集: 启用 packing 提高效率
   - 长文本数据集: 禁用 packing 避免干扰

5. 监控指标
   - Loss 应持续下降
   - 关注 eval_loss 防止过拟合
   - 保存多个 checkpoint 便于对比
""")

print("\n" + "=" * 70)
print("第三阶段完成！")
print("=" * 70)
