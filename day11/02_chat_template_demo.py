"""
第二阶段：指令数据与 Chat Template
==================================
理解数据格式如何决定模型行为
"""

# ============================================
# 第一部分：SFT 数据集格式对比
# ============================================

print("=" * 70)
print("【1】SFT 数据集格式对比：Alpaca vs ShareGPT")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  Alpaca 格式 (指令三元组)                                            │
├─────────────────────────────────────────────────────────────────────┤
│  {                                                                  │
│    "instruction": "解释什么是机器学习",                               │
│    "input": "请用简单的语言说明",                                     │
│    "output": "机器学习是一种让计算机从数据中学习的方法..."            │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ShareGPT 格式 (对话列表)                                            │
├─────────────────────────────────────────────────────────────────────┤
│  {                                                                  │
│    "conversations": [                                               │
│      {"from": "human", "value": "你好，请帮我写个故事"},              │
│      {"from": "gpt", "value": "好的，这是一个关于..."}               │
│    ]                                                                │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘

关键区别：
  - Alpaca: 单轮指令，适合基础能力训练
  - ShareGPT: 多轮对话，适合对话能力训练
""")

# ============================================
# 第二部分：Chat Template 核心概念
# ============================================

print("\n" + "=" * 70)
print("【2】Chat Template 核心痛点")
print("=" * 70)

print("""
问题：模型不知道谁在说话！

没有 Chat Template 的混乱输入：
  "你好，请帮我写个故事。好的，这是一个关于..."
  
模型困惑：
  - 这是用户说的还是助手说的？
  - 我应该继续生成还是停止？
  - 多轮对话如何区分角色？

解决方案：使用特殊 Token 标记角色
""")

# ============================================
# 第三部分：实战 - 使用 apply_chat_template
# ============================================

print("\n" + "=" * 70)
print("【3】实战：tokenizer.apply_chat_template")
print("=" * 70)

from transformers import AutoTokenizer

# 加载 TinyLlama 的 tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"\n加载模型: {model_name}")
print(f"Chat Template:\n{tokenizer.chat_template}")

# 定义对话
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "你好，请帮我写个故事。"},
    {"role": "assistant", "content": "好的，这是一个关于勇敢小兔子的故事..."},
    {"role": "user", "content": "这个故事很有趣，还有吗？"},
]

print("\n" + "-" * 50)
print("原始对话结构:")
for msg in messages:
    print(f"  [{msg['role']}] {msg['content'][:30]}...")

# 应用 chat template
print("\n" + "-" * 50)
print("应用 Chat Template 后:")
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)

# 对比：tokenize=True
print("\n" + "-" * 50)
print("Tokenize 后的结果:")
tokenized = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
print(f"  Token IDs shape: {tokenized.shape}")
print(f"  Token IDs: {tokenized[0][:20].tolist()}...")

# ============================================
# 第四部分：不同模型的 Chat Template 对比
# ============================================

print("\n" + "=" * 70)
print("【4】不同模型的 Chat Template 对比")
print("=" * 70)

# 测试不同模型的 template
models_to_test = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

for model in models_to_test:
    try:
        tok = AutoTokenizer.from_pretrained(model)
        print(f"\n模型: {model}")
        print("-" * 50)
        
        test_messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        ]
        
        formatted = tok.apply_chat_template(test_messages, tokenize=False)
        print(f"格式化输出:\n{formatted}")
        
    except Exception as e:
        print(f"\n模型: {model}")
        print(f"错误: {e}")

# ============================================
# 第五部分：加载中文 Alpaca 数据集
# ============================================

print("\n" + "=" * 70)
print("【5】加载 silk-road/alpaca-data-gpt4-chinese 数据集")
print("=" * 70)

from datasets import load_dataset

print("\n正在加载数据集...")
try:
    dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train[:10]")
    print(f"✓ 成功加载！样本数量: {len(dataset)}")
    print(f"特征列: {dataset.column_names}")
    
    # 查看第一个样本
    print("\n第一个样本:")
    sample = dataset[0]
    for key, value in sample.items():
        if value:
            print(f"  {key}: {str(value)[:80]}...")
        else:
            print(f"  {key}: (空)")
            
except Exception as e:
    print(f"✗ 加载失败: {e}")
    print("将使用模拟数据进行演示...")
    dataset = None

# ============================================
# 第六部分：数据格式转换
# ============================================

print("\n" + "=" * 70)
print("【6】数据格式转换：Alpaca → Chat Format")
print("=" * 70)

def alpaca_to_chat_format(example):
    """
    将 Alpaca 格式转换为 Chat 格式
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # 构建用户消息
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    
    return messages

# 使用真实数据或模拟数据
if dataset:
    sample_data = dataset[0]
else:
    # 模拟数据
    sample_data = {
        "instruction": "解释什么是机器学习",
        "input": "请用简单的语言说明",
        "output": "机器学习是一种让计算机从数据中学习的方法，不需要显式编程。"
    }

print("\n原始 Alpaca 格式:")
print(f"  instruction: {sample_data['instruction']}")
print(f"  input: {sample_data.get('input', '(空)')}")
print(f"  output: {sample_data['output'][:50]}...")

print("\n转换为 Chat 格式:")
chat_messages = alpaca_to_chat_format(sample_data)
for msg in chat_messages:
    print(f"  [{msg['role']}] {msg['content'][:60]}...")

print("\n应用 Chat Template:")
formatted_for_model = tokenizer.apply_chat_template(chat_messages, tokenize=False)
print(formatted_for_model)

# ============================================
# 第七部分：完整的数据处理流程
# ============================================

print("\n" + "=" * 70)
print("【7】完整的数据处理流程")
print("=" * 70)

def format_alpaca_dataset(examples, tokenizer):
    """
    批量处理 Alpaca 数据集，转换为模型输入格式
    """
    batch_messages = []
    
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
        output = examples["output"][i]
        
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        batch_messages.append(messages)
    
    # 应用 chat template
    formatted_texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in batch_messages
    ]
    
    return {"formatted_text": formatted_texts}

print("\n数据处理函数已定义")
print("使用方式:")
print("""
  # 批量处理数据集
  formatted_dataset = dataset.map(
      lambda x: format_alpaca_dataset(x, tokenizer),
      batched=True,
      batch_size=100
  )
  
  # 查看结果
  print(formatted_dataset[0]["formatted_text"])
""")

# ============================================
# 第八部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【8】关键要点总结")
print("=" * 70)

print("""
1. 数据格式决定模型行为
   - 不同的格式对应不同的训练目标
   - Chat 格式更适合对话场景

2. Chat Template 的作用
   - 明确区分 user/assistant/system 角色
   - 使用特殊 token 标记对话边界
   - 让模型知道何时开始/停止生成

3. 核心 API
   - tokenizer.apply_chat_template(messages, tokenize=False)
   - add_generation_prompt=True 会在末尾添加 assistant 标记

4. 数据处理流程
   - 加载原始数据 (Alpaca/ShareGPT)
   - 转换为统一的消息格式
   - 应用 Chat Template
   - Tokenize 并准备训练
""")

print("\n" + "=" * 70)
print("第二阶段完成！")
print("=" * 70)
