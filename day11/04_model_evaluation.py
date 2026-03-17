"""
第四阶段：效果评估与推理
========================
检验模型是否真的"听话"了
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================
# 第一部分：模型保存与加载
# ============================================

print("=" * 70)
print("【1】模型保存与加载")
print("=" * 70)

print("""
模型保存方式:
┌─────────────────────────────────────────────────────────────────────┐
│  方式 1: trainer.save_model()                                       │
│    trainer.save_model("./my_sft_model")                             │
│    tokenizer.save_pretrained("./my_sft_model")                      │
│                                                                     │
│  方式 2: 直接保存状态字典                                            │
│    torch.save(model.state_dict(), "model.pt")                       │
│                                                                     │
│  推荐: 方式 1，保存完整模型结构 + 权重 + Tokenizer                   │
└─────────────────────────────────────────────────────────────────────┘

模型加载方式:
┌─────────────────────────────────────────────────────────────────────┐
│  from transformers import AutoModelForCausalLM, AutoTokenizer       │
│                                                                     │
│  model = AutoModelForCausalLM.from_pretrained("./my_sft_model")     │
│  tokenizer = AutoTokenizer.from_pretrained("./my_sft_model")        │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：加载基座模型和 SFT 模型
# ============================================

print("\n" + "=" * 70)
print("【2】加载模型进行对比")
print("=" * 70)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
sft_model_path = "./sft_output/final_model"

print(f"\n基座模型: {model_name}")
print(f"SFT 模型路径: {sft_model_path}")

# 加载 tokenizer (两个模型使用相同的 tokenizer)
print("\n加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载基座模型
print("\n加载基座模型 (Base Model)...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print(f"✓ 基座模型加载完成")

# 尝试加载 SFT 模型
sft_model = None
if os.path.exists(sft_model_path):
    print("\n加载 SFT 模型...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"✓ SFT 模型加载完成")
else:
    print(f"\n⚠ SFT 模型未找到: {sft_model_path}")
    print("  将只测试基座模型")

# ============================================
# 第三部分：生成函数封装
# ============================================

print("\n" + "=" * 70)
print("【3】生成函数封装")
print("=" * 70)

def generate_with_model(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    """
    使用模型生成回复
    """
    # 应用 chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的回复
    if "<|assistant|>" in full_text:
        response = full_text.split("<|assistant|>")[-1].strip()
        # 去除 </s>
        response = response.replace("</s>", "").strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去除 prompt 部分
        if prompt in response:
            response = response.split(prompt)[-1].strip()
    
    return response

def generate_base_only(model, tokenizer, prompt, max_new_tokens=200):
    """
    基座模型生成（不使用 chat template，直接续写）
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去除 prompt
    if prompt in full_text:
        response = full_text.split(prompt)[-1].strip()
    else:
        response = full_text.strip()
    
    return response

print("生成函数已定义")

# ============================================
# 第四部分：微调前后对比测试
# ============================================

print("\n" + "=" * 70)
print("【4】微调前后'生死时速'对比")
print("=" * 70)

# 测试用例
test_cases = [
    {
        "name": "红烧肉菜谱",
        "prompt": "写一段关于红烧肉的菜谱",
        "description": "测试指令遵循能力"
    },
    {
        "name": "解释概念",
        "prompt": "用简单的语言解释什么是机器学习",
        "description": "测试知识表达能力"
    },
    {
        "name": "创意写作",
        "prompt": "写一首关于春天的短诗",
        "description": "测试创意生成能力"
    },
    {
        "name": "对话能力",
        "prompt": "你好，请介绍一下自己",
        "description": "测试对话交互能力"
    },
]

print("\n" + "=" * 70)
print("开始对比测试")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"测试 {i}: {test['name']}")
    print(f"描述: {test['description']}")
    print(f"{'='*70}")
    print(f"\n用户输入: {test['prompt']}")
    
    # 基座模型（不使用 chat template，直接续写）
    print("\n" + "-" * 50)
    print("【基座模型 - 直接续写模式】")
    print("-" * 50)
    base_response_direct = generate_base_only(base_model, tokenizer, test['prompt'])
    print(f"输出: {base_response_direct[:300]}...")
    
    # 基座模型（使用 chat template）
    print("\n" + "-" * 50)
    print("【基座模型 - Chat 模式】")
    print("-" * 50)
    base_response_chat = generate_with_model(base_model, tokenizer, test['prompt'])
    print(f"输出: {base_response_chat[:300]}...")
    
    # SFT 模型
    if sft_model:
        print("\n" + "-" * 50)
        print("【SFT 微调后模型】")
        print("-" * 50)
        sft_response = generate_with_model(sft_model, tokenizer, test['prompt'])
        print(f"输出: {sft_response[:300]}...")

# ============================================
# 第五部分：关键观察点
# ============================================

print("\n" + "=" * 70)
print("【5】关键观察点总结")
print("=" * 70)

print("""
对比维度:
┌─────────────────────────────────────────────────────────────────────┐
│  1. 指令遵循能力                                                     │
│     - 基座模型: 可能续写无关内容，或重复 prompt                      │
│     - SFT 模型: 准确理解并执行指令                                   │
│                                                                     │
│  2. 回复格式                                                         │
│     - 基座模型: 无固定格式，可能中途停止或无限续写                   │
│     - SFT 模型: 遵循 Chat Template，有明确的回答边界                 │
│                                                                     │
│  3. 内容相关性                                                       │
│     - 基座模型: 可能偏离主题，生成无关内容                           │
│     - SFT 模型: 紧扣主题，给出相关回答                               │
│                                                                     │
│  4. 对话连贯性                                                       │
│     - 基座模型: 不理解对话上下文                                     │
│     - SFT 模型: 能维持多轮对话的连贯性                               │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第六部分：交互式测试
# ============================================

print("\n" + "=" * 70)
print("【6】交互式测试")
print("=" * 70)

print("""
你可以输入自己的问题来测试模型。
输入 'quit' 退出测试。
""")

# 由于是非交互式环境，这里使用预设问题演示
demo_questions = [
    "如何学习编程？",
    "推荐几本好书",
]

print("\n预设问题测试:")
for question in demo_questions:
    print(f"\n用户: {question}")
    
    print("\n基座模型回答:")
    base_answer = generate_with_model(base_model, tokenizer, question, max_new_tokens=150)
    print(f"  {base_answer[:200]}...")
    
    if sft_model:
        print("\nSFT 模型回答:")
        sft_answer = generate_with_model(sft_model, tokenizer, question, max_new_tokens=150)
        print(f"  {sft_answer[:200]}...")

# ============================================
# 第七部分：模型评估指标
# ============================================

print("\n" + "=" * 70)
print("【7】模型评估指标")
print("=" * 70)

print("""
定量评估指标:
┌─────────────────────────────────────────────────────────────────────┐
│  1. Perplexity (困惑度)                                              │
│     - 衡量模型对测试数据的预测能力                                   │
│     - 越低越好                                                       │
│     - 代码: torch.exp(loss)                                          │
│                                                                     │
│  2. BLEU/ROUGE 分数                                                  │
│     - 与参考答案的相似度                                             │
│     - 需要标注好的测试集                                             │
│                                                                     │
│  3. 人工评估                                                         │
│     - 有用性 (Helpfulness)                                           │
│     - 相关性 (Relevance)                                             │
│     - 准确性 (Accuracy)                                              │
│     - 流畅性 (Fluency)                                               │
└─────────────────────────────────────────────────────────────────────┘

定性评估方法:
  - 准备 50-100 个覆盖不同场景的测试问题
  - 对比基座模型和 SFT 模型的回答
  - 人工打分或使用 GPT-4 辅助评估
""")

# ============================================
# 第八部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【8】效果评估关键要点总结")
print("=" * 70)

print("""
1. SFT 的核心价值
   - 让模型从"续写机器"变成"指令遵循助手"
   - 即使几亿参数的小模型也能有明显改善

2. 评估重点
   - 指令遵循能力 > 生成流畅度
   - 回答相关性 > 内容长度
   - 对话连贯性 > 单次回复质量

3. 常见失败模式
   - 过拟合: 模型只学会训练数据的特定格式
   - 灾难性遗忘: 丢失预训练时的通用知识
   - 幻觉: 生成看似合理但实际错误的内容

4. 迭代优化方向
   - 增加多样化训练数据
   - 调整学习率和训练轮数
   - 使用 LoRA 等参数高效微调方法
   - 引入 RLHF 进一步提升对齐度
""")

print("\n" + "=" * 70)
print("第四阶段完成！")
print("=" * 70)
