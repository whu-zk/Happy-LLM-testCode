from transformers import pipeline

# 加载 GPT-2 的文本生成pipeline
generator = pipeline('text-generation', model='gpt2')

prompt = "The capital of France is"
results = generator(prompt, max_new_tokens=10, num_return_sequences=3, pad_token_id=50256)

print("\n--- GPT-2 (Decoder-only) 续写结果 ---")
for i, res in enumerate(results):
    print(f"生成内容 {i+1}: {res['generated_text']}")
