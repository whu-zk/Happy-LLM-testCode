from transformers import pipeline

# 加载 BERT 的掩码填充pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')

text = "The capital of France is [MASK]."
results = unmasker(text)

print("--- BERT (Encoder-only) 预测结果 ---")
for res in results:
    print(f"预测词: {res['token_str']}, 置信度: {res['score']:.4f}")
