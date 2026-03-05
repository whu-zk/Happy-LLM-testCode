import matplotlib.pyplot as plt
import torch

def visualize_causal_mask(size):
    # torch.tril 生成下三角矩阵
    mask = torch.tril(torch.ones(size, size))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='Blues')
    plt.title("Look-ahead (Causal) Mask")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    
    # 添加文字解释
    for i in range(size):
        for j in range(size):
            text = "Keep" if mask[i, j] == 1 else "Mask"
            plt.text(j, i, text, ha="center", va="center", color="black" if mask[i,j] == 1 else "grey")
            
    plt.show()
    plt.savefig("./causal_mask.png")

visualize_causal_mask(size=5)
