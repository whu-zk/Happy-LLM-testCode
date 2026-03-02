import gensim
import numpy as np

def load_glove_vectors(glove_file_path):
    """
    加载 GloVe 词向量文件
    :param glove_file_path: GloVe 文件路径（如 glove.6B.100d.txt）
    :return: gensim的KeyedVectors对象（可直接调用词向量）
    """
    # 初始化词向量字典
    word_vectors = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 拆分每行：词 + 向量值
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    
    # 转换为Gensim的KeyedVectors（方便后续操作）
    vocab_size = len(word_vectors)
    vector_dim = len(next(iter(word_vectors.values())))
    kv = gensim.models.KeyedVectors(vector_dim)
    kv.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))
    return kv

# 示例：加载GloVe 6B 100维向量（需先下载：https://nlp.stanford.edu/projects/glove/）
glove_path = "./glove.6B.100d.txt"  # 替换为你的GloVe文件路径
wv = load_glove_vectors(glove_path)

# 测试加载结果：获取"king"的词向量
print("king的词向量前5维：", wv["king"][:5])
# 测试相似词：验证加载成功
print("\n与king最相似的5个词：")
print(wv.most_similar("king", topn=5))

# 验证数学公式: king - man + woman = ?
result = wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"\nKing - Man + Woman = {result[0][0]}")
