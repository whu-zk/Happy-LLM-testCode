"""
Hugging Face 生态系统大阅兵
===========================
演示四剑客的核心用法：transformers, datasets, accelerate, trl
"""

# ============================================
# 第一部分：transformers - AutoModel 的魔力
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print("=" * 60)
print("【1】AutoModel 自动识别模型架构")
print("=" * 60)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"\n加载模型: {model_name}")
print("-" * 40)

config = AutoConfig.from_pretrained(model_name)
print(f"\n模型配置信息:")
print(f"  - 架构类型: {config.architectures}")
print(f"  - 隐藏层维度: {config.hidden_size}")
print(f"  - 层数: {config.num_hidden_layers}")
print(f"  - 注意力头数: {config.num_attention_heads}")
print(f"  - 词汇表大小: {config.vocab_size}")

print("\n加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"  - Tokenizer 类型: {type(tokenizer).__name__}")
print(f"  - 特殊 token: PAD={tokenizer.pad_token}, EOS={tokenizer.eos_token}")

print("\n加载模型 (这可能需要一些时间)...")
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"  - 模型类型: {type(model).__name__}")
print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# ============================================
# 第二部分：Tokenizer 的工作原理
# ============================================

print("\n" + "=" * 60)
print("【2】Tokenizer 分词原理")
print("=" * 60)

text = "Hello, Hugging Face! 你好，世界！"
print(f"\n原始文本: {text}")

tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")

input_ids = tokenizer.encode(text)
print(f"Token IDs: {input_ids}")

decoded = tokenizer.decode(input_ids)
print(f"解码还原: {decoded}")

# ============================================
# 第三部分：简单推理示例
# ============================================

print("\n" + "=" * 60)
print("【3】简单推理示例")
print("=" * 60)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"\n输入: {prompt}")
print(f"编码后 shape: {inputs['input_ids'].shape}")

print("\n生成中...")
output_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"生成结果: {generated_text}")

# ============================================
# 第四部分：datasets 数据处理
# ============================================

print("\n" + "=" * 60)
print("【4】datasets - 高效数据处理")
print("=" * 60)

from datasets import load_dataset

print("\n加载示例数据集 (小规模测试)...")
try:
    dataset = load_dataset("imdb", split="train[:5]")
    print(f"  - 数据集名称: imdb")
    print(f"  - 样本数量: {len(dataset)}")
    print(f"  - 特征列: {dataset.column_names}")
    print(f"\n第一个样本:")
    print(f"  - 文本 (前100字): {dataset[0]['text'][:100]}...")
    print(f"  - 标签: {'正面' if dataset[0]['label'] == 1 else '负面'}")
except Exception as e:
    print(f"  加载失败 (可能需要网络): {e}")

# ============================================
# 第五部分：accelerate 设备切换
# ============================================

print("\n" + "=" * 60)
print("【5】accelerate - 一行代码切换设备")
print("=" * 60)

import torch
from accelerate import Accelerator

accelerator = Accelerator()
print(f"\n当前设备信息:")
print(f"  - 设备类型: {accelerator.device.type}")
print(f"  - 可用 GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# 第六部分：trl 简介
# ============================================

print("\n" + "=" * 60)
print("【6】trl - SFT 和 RLHF 进阶库")
print("=" * 60)

print("""
trl (Transformer Reinforcement Learning) 提供了:

1. SFTTrainer - 监督微调训练器
   - 简化指令微调流程
   - 支持 LoRA 等高效微调方法

2. PPOTrainer - 强化学习训练器
   - 用于 RLHF (人类反馈强化学习)
   - 实现PPO算法对齐模型

3. DPOTrainer - 直接偏好优化
   - 无需奖励模型的偏好学习
   - 更稳定的训练过程

示例用法:
    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text"
    )
    trainer.train()
""")

print("\n" + "=" * 60)
print("演示完成！")
print("=" * 60)
