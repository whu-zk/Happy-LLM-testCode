import sentencepiece as sp

# 模拟一个简单的中英文混合语料
data = """
I love deep learning. 我爱深度学习。
Large language models are amazing. 大语言模型太神奇了。
Transformer is the core of LLaMA. Transformer 是 LLaMA 的核心。
""" * 1000  # 复制多次模拟大数据量

with open("./train.txt", "w", encoding="utf-8") as f:
    f.write(data)


# 训练模型
sp.SentencePieceTrainer.train(
    input='train.txt',           # 输入文件
    model_prefix='my_tokenizer', # 输出模型文件名前缀
    vocab_size=483,             # 词表大小（实战通常为32000，此处设小方便演示）
    character_coverage=0.9995,   # 覆盖 99.95% 的字符，适合中英文
    model_type='bpe',            # 使用 BPE 算法
    user_defined_symbols=['<pad>', '<mask>'], # 自定义特殊符号
    byte_fallback=True           # 遇到不认识的字符回退到字节表示，永不报错
)

print("训练完成！已生成 my_tokenizer.model 和 my_tokenizer.vocab")

# 加载模型
sp_model = sp.SentencePieceProcessor(model_file='my_tokenizer.model')

# 测试用例
text_en = "I love deep learning"
text_cn = "我爱深度学习"

# 1. 编码 (Text -> IDs)
ids_en = sp_model.encode_as_ids(text_en)
ids_cn = sp_model.encode_as_ids(text_cn)

# 2. 分词结果 (Text -> Tokens)
tokens_en = sp_model.encode_as_pieces(text_en)
tokens_cn = sp_model.encode_as_pieces(text_cn)

print(f"英文分词: {tokens_en}")
print(f"英文 ID: {ids_en}")
print(f"\n中文分词: {tokens_cn}")
print(f"中文 ID: {ids_cn}")

# 3. 解码 (IDs -> Text)
print(f"\n解码测试: {sp_model.decode(ids_cn)}")
