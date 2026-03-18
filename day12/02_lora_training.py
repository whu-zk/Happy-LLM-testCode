"""
第二阶段：LoRA 微调实战
=====================
使用 PEFT 库进行参数高效微调
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
import os

# ============================================
# 第一部分：LoRA 配置详解
# ============================================

print("=" * 70)
print("【1】LoRA 配置详解")
print("=" * 70)

print("""
LoRA 核心参数:
┌─────────────────────────────────────────────────────────────────────┐
│  r (rank): LoRA 的秩                                                 │
│    - 决定低秩矩阵的维度                                              │
│    - 常用值: 8, 16, 32, 64                                           │
│    - 越大表达能力越强，但参数量也越多                                │
│    - 推荐: 小模型(1-3B)用 8-16，大模型(7B+)用 16-64                  │
│                                                                     │
│  lora_alpha: 缩放参数                                                │
│    - 控制 LoRA 更新的强度                                            │
│    - 通常设为 r 的 2 倍 (如 r=16, alpha=32)                          │
│    - 公式: 实际缩放 = alpha / r                                      │
│                                                                     │
│  target_modules: 目标模块                                            │
│    - 指定对哪些层应用 LoRA                                           │
│    - 常见选择:                                                       │
│      * ["q_proj", "v_proj"]: 只对 attention 的 Q,V 投影             │
│      * ["q_proj", "k_proj", "v_proj", "o_proj"]: 全部 attention      │
│      * 包含 "gate_proj", "up_proj", "down_proj": 加上 FFN            │
│    - 越多模块效果越好，但参数量也越多                                │
│                                                                     │
│  lora_dropout: Dropout 率                                            │
│    - 防止过拟合                                                      │
│    - 常用值: 0.0 - 0.1                                               │
│                                                                     │
│  bias: 偏置训练方式                                                  │
│    - "none": 不训练偏置                                              │
│    - "all": 训练所有偏置                                             │
│    - "lora_only": 只训练 LoRA 层的偏置                               │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：加载模型
# ============================================

print("\n" + "=" * 70)
print("【2】加载模型并应用 LoRA")
print("=" * 70)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./lora_output"

print(f"\n加载模型: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print(f"模型加载完成")
print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============================================
# 第三部分：配置并应用 LoRA
# ============================================

print("\n" + "=" * 70)
print("【3】配置并应用 LoRA")
print("=" * 70)

# LoRA 配置
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
    task_type=TaskType.CAUSAL_LM,   # 任务类型
)

# 应用 LoRA
print("\n应用 LoRA...")
model = get_peft_model(model, lora_config)

# 修复 gradient checkpointing 与 LoRA 的兼容性问题
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# 打印可训练参数
print("\n" + "-" * 50)
model.print_trainable_parameters()
print("-" * 50)

# 计算显存节省
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_ratio = trainable_params / total_params * 100

print(f"\n显存节省估算:")
print(f"  全参数训练显存: ~{total_params * 20 / (1024**3):.1f} GB")
print(f"  LoRA 训练显存: ~{total_params * 2 / (1024**3) + trainable_params * 16 / (1024**3):.1f} GB")
print(f"  节省比例: {(1 - (total_params * 2 + trainable_params * 16) / (total_params * 20)) * 100:.1f}%")

# ============================================
# 第四部分：准备数据
# ============================================

print("\n" + "=" * 70)
print("【4】准备训练数据")
print("=" * 70)

# 加载数据集
print("\n加载数据集: silk-road/alpaca-data-gpt4-chinese")
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:100]")
print(f"  - 样本数量: {len(dataset)}")

# 数据格式化
def format_alpaca_to_chat(example):
    """将 Alpaca 格式转换为 Chat 格式"""
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
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

# 格式化数据集
print("\n格式化数据集...")
formatted_dataset = dataset.map(format_alpaca_to_chat)
formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col != "text"]
)

print(f"\n格式化后示例:")
print("-" * 50)
print(formatted_dataset[0]["text"][:300])
print("-" * 50)

# ============================================
# 第五部分：配置训练参数
# ============================================

print("\n" + "=" * 70)
print("【5】配置训练参数")
print("=" * 70)

sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,      # LoRA 可以用更大的 batch
    gradient_accumulation_steps=2,       # 等效 batch=4
    learning_rate=2e-4,                  # LoRA 可以用更大的学习率
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    optim="adamw_torch",
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    # SFT 特有
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
)

print("训练参数:")
print(f"  - Epochs: {sft_config.num_train_epochs}")
print(f"  - Batch size: {sft_config.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {sft_config.gradient_accumulation_steps}")
print(f"  - Learning rate: {sft_config.learning_rate}")
print(f"  - LoRA rank: {lora_config.r}")
print(f"  - LoRA alpha: {lora_config.lora_alpha}")

# ============================================
# 第六部分：启动训练
# ============================================

print("\n" + "=" * 70)
print("【6】启动 LoRA 训练")
print("=" * 70)

# 初始化 trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=sft_config,
)

print("\n开始训练...")
print("(按 Ctrl+C 可中断训练)\n")

try:
    trainer.train()
    
    # 保存 LoRA 权重
    print("\n保存 LoRA 权重...")
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    print(f"✓ LoRA 权重已保存到: {output_dir}/lora_adapter")
    
except KeyboardInterrupt:
    print("\n\n训练被中断")
    print("保存当前进度...")
    model.save_pretrained(os.path.join(output_dir, "lora_adapter_interrupted"))
    print(f"✓ LoRA 权重已保存到: {output_dir}/lora_adapter_interrupted")

# ============================================
# 第七部分：LoRA 权重合并与推理
# ============================================

print("\n" + "=" * 70)
print("【7】LoRA 权重合并与推理")
print("=" * 70)

print("""
LoRA 权重使用方式:
┌─────────────────────────────────────────────────────────────────────┐
│  方式 1: 动态加载 (推荐用于多任务切换)                               │
│    from peft import PeftModel                                        │
│    base_model = AutoModelForCausalLM.from_pretrained(base_path)      │
│    model = PeftModel.from_pretrained(base_model, lora_path)          │
│                                                                     │
│  方式 2: 合并权重 (推荐用于部署)                                     │
│    model = model.merge_and_unload()  # 合并 LoRA 到基座              │
│    model.save_pretrained(merged_path)                                │
└─────────────────────────────────────────────────────────────────────┘
""")

# 测试生成
def generate_response(model, tokenizer, prompt, max_length=200):
    """生成回复"""
    messages = [{"role": "user", "content": prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip().replace("</s>", "")
    
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
print("【8】LoRA 微调关键要点总结")
print("=" * 70)

print("""
1. LoRA 优势
   - 显存节省 80%+ (7B 模型从 150GB → 25GB)
   - 只训练 0.1%-1% 的参数
   - 不增加推理延迟 (可合并权重)
   - 效果接近全参数微调

2. 参数选择建议
   - r=8/16: 消费级 GPU (16-24GB)
   - r=32/64: 专业级 GPU (40-80GB)
   - alpha=2*r: 标准配置
   - target_modules: 至少 ["q_proj", "v_proj"]

3. 学习率调整
   - LoRA 可以用更大的学习率 (2e-4 vs 2e-5)
   - 因为更新的参数少，不容易发散

4. 多任务管理
   - 不同任务训练不同的 LoRA 权重
   - 运行时动态切换，无需重新加载基座模型
   - 适合构建多角色/多领域对话系统

5. 部署优化
   - 合并权重后推理速度与基座模型相同
   - 也可单独保存 LoRA 权重，按需加载
""")

print("\n" + "=" * 70)
print("第二阶段完成！")
print("=" * 70)
