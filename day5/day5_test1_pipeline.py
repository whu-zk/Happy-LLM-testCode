from transformers import pipeline

# 1. 情感分析 (分类任务)
classifier = pipeline("sentiment-analysis")
print(classifier("This course is amazing!"))

# 2. 文本生成 (生成任务)
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time, AI", max_new_tokens=10, pad_token_id=50256))
