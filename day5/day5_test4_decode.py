import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# 1. 加载模型和分词器
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, max_new_tokens=20, temperature=1.0):
    # 将输入文本转换为 Tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    print(f"\n[Prompt]: {prompt}")
    print("-" * 30)
    print(prompt, end="", flush=True)

    # 循环生成每一个词
    for _ in range(max_new_tokens):
        # 得到模型的输出 (Logits)
        with torch.no_grad():
            outputs = model(input_ids)
            # 取最后一个词的预测结果 [batch, seq_len, vocab_size] -> [vocab_size]
            next_token_logits = outputs.logits[0, -1, :]
            
            # 应用温度参数
            next_token_logits = next_token_logits / temperature
            
            # 使用 Softmax 转换为概率
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 根据概率分布采样一个词 ID
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # 将新词 ID 拼接到输入序列中，用于下一次预测
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            
            # 解码并打印这个词
            output_word = tokenizer.decode(next_token_id)
            print(output_word, end="", flush=True)

# --- 实验对比 ---

# 实验 1：低温度 (逻辑严谨但可能枯燥)
print("\n\nExperiment 1: Temperature = 0.1 (Conservative)")
generate_text("The future of AI is", temperature=0.1)

# 实验 2：高温度 (充满想象力但可能混乱)
print("\n\nExperiment 2: Temperature = 1.5 (Creative/Wild)")
generate_text("The future of AI is", temperature=1.5)
