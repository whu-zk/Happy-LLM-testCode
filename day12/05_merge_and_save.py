"""
第五阶段：权重合并与保存
=======================
学习如何将 LoRA 适配器合并到基座模型，生成完整的新模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import os

# ============================================
# 第一部分：理解 LoRA 适配器
# ============================================

print("=" * 70)
print("【1】理解 LoRA 适配器")
print("=" * 70)

print("""
LoRA 训练后的文件结构:
┌─────────────────────────────────────────────────────────────────────┐
│  基座模型 (Base Model)                                               │
│  - 大小: 7B 模型约 14GB (FP16)                                       │
│  - 位置: Hugging Face Hub 或本地路径                                 │
│                                                                     │
│  LoRA 适配器 (Adapter)                                               │
│  - 大小: 仅几十 MB (如 16MB)                                         │
│  - 包含: LoRA 权重 (A, B 矩阵) + 配置                                │
│  - 位置: ./lora_output/adapter/                                      │
└─────────────────────────────────────────────────────────────────────┘

为什么适配器这么小？
┌─────────────────────────────────────────────────────────────────────┐
│  7B 模型参数: 7,000,000,000                                          │
│  LoRA 参数 (r=16): ~4,500,000 (0.06%)                                │
│                                                                     │
│  存储对比:                                                           │
│    - 基座模型: 14,000 MB                                             │
│    - LoRA 适配器: ~16 MB                                             │
│    - 比例: 1:875                                                     │
└─────────────────────────────────────────────────────────────────────┘

两种使用方式:
┌─────────────────────────────────────────────────────────────────────┐
│  方式 1: 动态加载 (多任务切换)                                        │
│    - 基座模型 + LoRA A → 任务 A                                      │
│    - 基座模型 + LoRA B → 任务 B                                      │
│    - 优点: 灵活切换，节省存储                                        │
│    - 缺点: 推理时需要加载两个文件                                    │
│                                                                     │
│  方式 2: 合并导出 (部署使用)                                          │
│    - 基座模型 + LoRA → 合并后的完整模型                              │
│    - 优点: 单一文件，推理更快                                        │
│    - 缺点: 每个任务需要单独存储完整模型                              │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================
# 第二部分：模拟训练一个 LoRA 适配器
# ============================================

print("\n" + "=" * 70)
print("【2】模拟训练并保存 LoRA 适配器")
print("=" * 70)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./demo_adapter"
merged_path = "./demo_merged_model"

print(f"\n加载基座模型: {model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ 基座模型加载完成")
print(f"  - 参数量: {sum(p.numel() for p in base_model.parameters()) / 1e9:.2f}B")

# 配置 LoRA
print("\n配置并应用 LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 保存适配器
print(f"\n保存 LoRA 适配器到: {adapter_path}")
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

# 计算文件大小
adapter_size = sum(
    os.path.getsize(os.path.join(adapter_path, f))
    for f in os.listdir(adapter_path)
    if os.path.isfile(os.path.join(adapter_path, f))
)
print(f"✓ 适配器大小: {adapter_size / 1024 / 1024:.2f} MB")

# ============================================
# 第三部分：权重合并详解
# ============================================

print("\n" + "=" * 70)
print("【3】权重合并详解")
print("=" * 70)

print("""
合并原理:
┌─────────────────────────────────────────────────────────────────────┐
│  数学公式: W_merged = W_base + ΔW = W_base + B × A                   │
│                                                                     │
│  合并过程:                                                           │
│    1. 加载基座模型权重 W_base                                        │
│    2. 加载 LoRA 权重 B, A                                            │
│    3. 计算 ΔW = B × A                                                │
│    4. 更新 W_base = W_base + ΔW                                      │
│    5. 移除 LoRA 结构，保存合并后的模型                               │
│                                                                     │
│  效果:                                                               │
│    - 模型结构变回普通模型 (不再是 PeftModel)                         │
│    - 推理速度与基座模型相同                                          │
│    - 不需要额外的 LoRA 计算开销                                      │
└─────────────────────────────────────────────────────────────────────┘

合并 vs 不合并对比:
┌─────────────────┬─────────────────┬─────────────────┐
│     特性        │   未合并        │    已合并       │
├─────────────────┼─────────────────┼─────────────────┤
│  文件数量       │   2个           │    1个          │
│  推理速度       │   稍慢          │    最快         │
│  显存占用       │   稍高          │    最低         │
│  多任务切换     │   灵活          │    不方便       │
│  部署复杂度     │   需要适配器    │    单一文件     │
└─────────────────┴─────────────────┴─────────────────┘
""")

# ============================================
# 第四部分：实战合并权重
# ============================================

print("\n" + "=" * 70)
print("【4】实战：合并 LoRA 权重")
print("=" * 70)

print("\n>>> 代码实现：")
print("-" * 50)
print("""
from peft import PeftModel

# 方式 1: 从保存的适配器加载并合并
base_model = AutoModelForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # 合并并卸载 LoRA 结构

# 方式 2: 直接合并内存中的模型 (如果已经训练好)
model = model.merge_and_unload()
""")
print("-" * 50)

# 实际合并
print("\n执行合并操作...")
print("  1. 调用 merge_and_unload()")

merged_model = model.merge_and_unload()

print("✓ 合并完成")
print(f"  - 模型类型: {type(merged_model).__name__}")
print(f"  - 参数量: {sum(p.numel() for p in merged_model.parameters()) / 1e9:.2f}B")

# 验证合并效果
print("\n验证合并效果:")
print("  - 检查是否为 PeftModel:", "PeftModel" in str(type(merged_model)))
print("  - 所有参数都可训练:", all(p.requires_grad for p in merged_model.parameters()))

# ============================================
# 第五部分：保存合并后的模型
# ============================================

print("\n" + "=" * 70)
print("【5】保存合并后的完整模型")
print("=" * 70)

print(f"\n保存合并模型到: {merged_path}")
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

# 计算文件大小
merged_size = sum(
    os.path.getsize(os.path.join(merged_path, f))
    for f in os.listdir(merged_path)
    if os.path.isfile(os.path.join(merged_path, f))
)
print(f"✓ 合并模型大小: {merged_size / 1024 / 1024:.2f} MB")

print(f"\n文件对比:")
print(f"  - LoRA 适配器: {adapter_size / 1024 / 1024:.2f} MB")
print(f"  - 合并后模型: {merged_size / 1024 / 1024:.2f} MB")
print(f"  - 大小比例: {merged_size / adapter_size:.0f}:1")

# ============================================
# 第六部分：加载与使用
# ============================================

print("\n" + "=" * 70)
print("【6】加载与使用合并后的模型")
print("=" * 70)

print("\n>>> 代码实现：")
print("-" * 50)
print("""
# 加载合并后的模型 (与普通模型完全相同)
model = AutoModelForCausalLM.from_pretrained(merged_path)
tokenizer = AutoTokenizer.from_pretrained(merged_path)

# 推理 (无需处理 LoRA)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
""")
print("-" * 50)

# 实际加载测试
print("\n测试加载合并后的模型...")
loaded_model = AutoModelForCausalLM.from_pretrained(
    merged_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
loaded_tokenizer = AutoTokenizer.from_pretrained(merged_path)

print("✓ 模型加载成功")

# 测试生成
def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n测试生成:")
prompt = "你好"
result = generate_text(loaded_model, loaded_tokenizer, prompt)
print(f"  Prompt: {prompt}")
print(f"  Output: {result[:100]}...")

# ============================================
# 第七部分：多适配器管理
# ============================================

print("\n" + "=" * 70)
print("【7】多适配器管理策略")
print("=" * 70)

print("""
场景：一个基座模型 + 多个 LoRA 适配器
┌─────────────────────────────────────────────────────────────────────┐
│  基座模型: meta-llama/Llama-2-7b-hf                                  │
│                                                                     │
│  适配器 A: ./adapters/coding/      → 代码生成任务                    │
│  适配器 B: ./adapters/chat/        → 对话任务                        │
│  适配器 C: ./adapters/summarize/   → 摘要任务                        │
└─────────────────────────────────────────────────────────────────────┘

动态切换代码:
┌─────────────────────────────────────────────────────────────────────┐
│  from peft import PeftModel                                          │
│                                                                     │
│  # 加载基座                                                          │
│  base_model = AutoModelForCausalLM.from_pretrained("llama-2-7b")     │
│                                                                     │
│  # 切换到代码生成任务                                                │
│  model = PeftModel.from_pretrained(base_model, "./adapters/coding")  │
│                                                                     │
│  # 切换到对话任务 (无需重新加载基座)                                 │
│  model.load_adapter("./adapters/chat", adapter_name="chat")          │
│  model.set_adapter("chat")                                           │
└─────────────────────────────────────────────────────────────────────┘

部署策略对比:
┌──────────────┬──────────────────────────────────────────────────────┐
│  开发阶段    │  使用动态加载，方便快速迭代测试不同适配器            │
├──────────────┼──────────────────────────────────────────────────────┤
│  生产部署    │  合并后部署，获得最佳推理性能                          │
├──────────────┼──────────────────────────────────────────────────────┤
│  多租户服务  │  基座模型常驻内存，按需加载不同适配器                  │
└──────────────┴──────────────────────────────────────────────────────┘
""")

# ============================================
# 第八部分：讨论 - LoRA vs 全参数微调
# ============================================

print("\n" + "=" * 70)
print("【8】讨论：LoRA vs 全参数微调")
print("=" * 70)

print("""
既然 LoRA 效果这么好，为什么还要全参数微调？

┌─────────────────────────────────────────────────────────────────────┐
│  LoRA 的优势 (为什么首选 LoRA)                                       │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ 显存效率高                                                       │
│     - 7B 模型: 150GB → 25GB (LoRA) → 8GB (QLoRA)                   │
│     - 消费级 GPU 也能微调大模型                                      │
│                                                                     │
│  ✅ 训练速度快                                                       │
│     - 只更新少量参数，反向传播更快                                   │
│     - 收敛通常更快                                                   │
│                                                                     │
│  ✅ 多任务灵活                                                       │
│     - 一个基座 + 多个适配器                                          │
│     - 轻松切换不同任务                                               │
│                                                                     │
│  ✅ 防止灾难性遗忘                                                   │
│     - 基座知识保持较好                                               │
│     - 不容易过拟合到训练数据                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  全参数微调的优势 (什么时候用全参数)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ✅ 表达能力更强                                                     │
│     - 可以修改任何参数                                               │
│     - 对于复杂任务可能效果更好                                       │
│                                                                     │
│  ✅ 领域适配彻底                                                     │
│     - 需要模型完全适应新领域                                         │
│     - 如医学、法律等专业领域                                         │
│                                                                     │
│  ✅ 预训练续训                                                       │
│     - 在特定领域数据上继续预训练                                     │
│     - 需要修改所有参数                                               │
│                                                                     │
│  ✅ 硬件充足时                                                       │
│     - A100/H100 等高端 GPU                                           │
│     - 追求极限性能                                                   │
└─────────────────────────────────────────────────────────────────────┘

实际选择建议:
┌─────────────────────────────────────────────────────────────────────┐
│  场景                          │  推荐方法                          │
├─────────────────────────────────┼────────────────────────────────────┤
│  消费级 GPU (8-24GB)           │  QLoRA / LoRA                      │
│  快速原型验证                  │  LoRA                              │
│  多任务服务                    │  LoRA (动态切换)                   │
│  生产部署 (单一任务)           │  LoRA 合并后部署                   │
│  领域深度适配 (医学/法律)      │  全参数微调                        │
│  预训练续训                    │  全参数微调                        │
│  追求极限性能                  │  全参数微调                        │
└─────────────────────────────────┴────────────────────────────────────┘

结论:
  LoRA 是 80% 场景的最佳选择！
  只有当 LoRA 效果不达标且硬件充足时，才考虑全参数微调。
""")

# ============================================
# 第九部分：关键要点总结
# ============================================

print("\n" + "=" * 70)
print("【9】权重合并关键要点总结")
print("=" * 70)

print("""
1. LoRA 适配器特点
   - 大小仅几十 MB (基座模型的 1/1000)
   - 包含 LoRA 权重 + 配置
   - 需要配合基座模型使用

2. 合并操作
   model = model.merge_and_unload()
   - 将 LoRA 权重融入基座
   - 生成单一完整模型
   - 推理速度最优

3. 使用场景
   - 开发阶段: 动态加载适配器
   - 生产部署: 合并后单一模型
   - 多任务: 基座 + 多个适配器

4. 保存方式
   - 适配器: model.save_pretrained(adapter_path)
   - 合并模型: merged_model.save_pretrained(merged_path)

5. 选择建议
   - 首选 LoRA/QLoRA (80% 场景)
   - 全参数微调仅用于深度领域适配
""")

print("\n" + "=" * 70)
print("第五阶段完成！")
print("=" * 70)
