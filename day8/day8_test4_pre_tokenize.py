import numpy as np
import sentencepiece as spm
from tqdm import tqdm

def pre_tokenize():
    # 1. 加载我们在 Phase 3 训练好的模型
    sp = spm.SentencePieceProcessor(model_file='my_tokenizer.model')
    
    # 2. 读取原始语料 (假设是一个很大的 txt)
    with open("train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    all_ids = []
    for line in tqdm(lines, desc="Tokenizing"):
        # 分词并转为 ID
        ids = sp.encode_as_ids(line)
        # 加上句子结束符 (EOS)
        ids.append(sp.eos_id())
        all_ids.extend(ids)
    
    # 3. 转换为 Numpy 数组并保存为二进制文件
    all_ids = np.array(all_ids, dtype=np.uint16) # 词表<65535用uint16即可
    all_ids.tofile("train.bin")
    print(f"\n预处理完成！总共 {len(all_ids)} 个 Tokens 已存入 train.bin")

pre_tokenize()
