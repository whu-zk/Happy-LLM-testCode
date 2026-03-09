# Happy-LLM-testCode & 学习计划
## 目录

- [Day 1：NLP 概览与文本表示](#day1)
- [Day 2：核心组件——注意力机制](#day2)
- [Day 3：Transformer 架构拆解（上）——结构细节](#day3)
- [Day 4：Transformer 架构拆解（下）——整体组装](#day4)
- [Day 5：预训练语言模型 (PLM) 的范式演进](#day5)

---

<a id="day1"></a>

# 📅 Day 1：NLP 概览与文本表示 —— 开启 LLM 之门

## 第一阶段：理论铺垫

**目标：了解 NLP 的前世今生，明确 LLM 的定位。**

### 1. NLP 的进化史
- 规则派 vs 统计派：理解为什么早期的翻译软件（如 2000 年代的谷歌翻译）效果很差，而现在的模型效果惊人。
- 语言模型 (Language Model) 的定义：核心公式 $P(w_n | w_1, w_2, ..., w_{n-1})$ 。理解 LLM 本质上就是一个“对下一个token的预测器”。

### 2. LLM 的独特性
- 规模效应 (Scaling Law)：为什么参数量从亿级跳到千亿级会发生质变？
- 阅读资料：阅读 Happy-LLM 文档第 1 章相关内容。

### 3. Checkpoint
- 思考：如果让你设计一个判断句子情感的模型，在没有深度学习前，你会怎么做？（提示：词典匹配）。

---

## 第二阶段：分词技术——文本的最小单元

**目标：掌握文本如何切分为模型可处理的 Token。**

### 1. 分词 (Tokenization) 的三种粒度
- Word-level：按空格分词（词表太大，无法处理未登录词）。
- Char-level：按字母分词（单个字母缺乏语义信息）。
- Subword-level (主流)：BPE (Byte Pair Encoding)、WordPiece。理解为什么 "unhappily" 会被切分为 "un" + "happi" + "ly"。

### 2. 词表 (Vocabulary) 的秘密
- 理解 Special Tokens：[CLS] (分类标志), [SEP] (分隔符), [PAD] (填充), [MASK] (掩码)。
- 词表大小 (Vocab Size) 对模型性能和计算量的影响。

### 3. 动手小实验
- 使用 Hugging Face 的 AutoTokenizer 加载 GPT-2 或 LLaMA 的分词器。
- 代码尝试：输入一段中文或英文，观察它被切分成了哪些 ID。

---

## 第三阶段：向量化——语义的数学表达

**目标：理解词向量（Embedding）如何捕捉语义相似性。**

### 1. 离散表示的困局
- One-hot Encoding：手画一个 10000 维的向量。理解为什么它无法表示“猫”和“狗”的相似性（正交性）。
- 维度灾难：词表越大，向量越稀疏，计算越低效。

### 2. 分布式表示与 Word2Vec
- 核心思想：上下文相似的词，其语义也相似（Distributional Hypothesis）。
- Word2Vec 原理：Skip-gram 与 CBOW。理解模型是如何通过预测周围词来学习到一个稠密向量（Embedding）的。

### 3. 语义空间
- 余弦相似度 (Cosine Similarity)：如何用数学计算两个词的距离。
- 词向量运算：理解著名的例子：King - Man + Woman = Queen。

---

## 第四阶段：动手实战

**目标：通过 Python 代码实现文本到向量的转换。**

### 1. 环境准备
- 安装 torch, transformers, scikit-learn。

### 2. 任务 A：手写 One-hot 编码
- 给定三个句子，手动构建词表，并输出每个词的 One-hot 向量。

### 3. 任务 B：使用预训练 Embedding 计算相似度
- 调用 gensim 库加载预训练的 Glove 模型。
- 寻找与“king”最接近的 5 个词。
- 可视化：使用 PCA 或 t-SNE 将高维词向量降维到 2D 平面并画图，观察近义词是否聚在一起。

---

## 第五阶段：复盘与预习

### 1. 知识梳理
- 今天学到了：分词 -> 词表索引 (ID) -> 词向量 (Embedding)。
- 核心结论：所有的文字在进入 LLM 之前，必须变成一组连续的浮点数向量。

### 2. 问题思考
- Word2Vec 是静态的（“苹果”在任何语境下向量都一样）。如果我想区分“我吃了一个苹果”和“苹果发布了新手机”，该怎么办？（这是 Day 2 注意力机制要解决的问题）。

### 3. 明日预告
- 预习：什么是“注意力机制”？尝试通俗理解 Q、K、V 是什么。

---

## 🛠 Day 1 必备工具包

- 在线阅读：Happy-LLM 第一章
- 课程重点：语言模型的定义、文本如何通过分词切分为模型可处理的 Token、词向量从离散表示到分布式表示的演进
- 推荐视频：3Blue1Brown 的《Transformer 视觉解说》入

---

<a id="day2"></a>

# 📅 Day 2：核心组件——注意力机制 (Attention Mechanism)

## 第一阶段：缘起与进化

**目标：理解从“死板”的词向量到“动态”的上下文表示的必要性。**

### 1. RNN 的痛点
- 长距离依赖问题：为什么句子太长，开头的词就被模型忘了？
- 串行计算瓶颈：为什么 RNN 训练慢？（必须等前一个词算完才能算下一个）。

### 2. 注意力机制的直观类比
- 视觉注意力：看一张图片时，你的眼睛会聚焦在某个点。
- 语言注意力：在句子“他把书还给了小明，因为他很慷慨”中，“他”到底指谁？模型需要通过“注意力”将“他”与“小明”关联。

### 3. 阅读资料
- 阅读 Happy-LLM 第 2 章开头：自注意力机制的引入。
- 思考：Word2Vec 中的“苹果”向量是固定的，但 Attention 后的“苹果”向量在不同句子里是否应该变？

---

## 第二阶段：深度解码 Q、K、V 机制

**目标：攻克 Transformer 的数学核心——缩放点积注意力。**

### 1. 三大矩阵的本质意义
- Query (查询 Q)：我想找什么？（当前词发出的询问）。
- Key (键 K)：我有什么？（其他词提供的索引信息）。
- Value (值 V)：我真正的内容是什么？（匹配成功后要提取的信息）。

### 2. 计算四步走
- 第一步：点积 (Dot-product)。计算 $Q$ 和 $K$ 的相关性得分。
- 第二步：缩放 (Scale)。为什么要除以 $\sqrt{d_k}$？（防止梯度消失/爆炸，保持 Softmax 稳定）。
- 第三步：归一化 (Softmax)。将得分变成概率分布（加起来等于 1）。
- 第四步：加权求和。用概率乘 $V$，得到最终的上下文表示。

### 3. 数学公式推导

$$
Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- 核心练习：手动在纸上模拟 2x2 矩阵的 Attention 计算过程。

---

## 第三阶段：多头注意力 (Multi-Head Attention)

**目标：理解“多头”如何让模型具备多维度的观察能力。**

### 1. 为什么要“多头”？
- 类比：一个人看句子（关注语法），另一个人看句子（关注情感）。
- 多个头可以同时捕捉不同的语义特征。

### 2. 多头的实现细节
- 拆分 (Split)：将高维向量切分成多个低维小向量。
- 并行计算：每个头独立计算 Attention。
- 拼接 (Concat)：把结果拼回来。
- 线性投影 (Linear)：最后再过一个全连接层进行整合。

### 3. Checkpoint
- 如果隐藏层维度是 512，设置 8 个头，每个头的维度是多少？

---

## 第四阶段：手写代码实战

**目标：用 PyTorch 从零实现 Scaled Dot-Product Attention。**

### 1. 任务 A：实现基础 Attention 函数
- 编写一个函数 `scaled_dot_product_attention(q, k, v)`。
- 要求：包含掩码 (Mask) 处理逻辑。

### 2. 任务 B：封装 Multi-Head Attention 类
- 继承 `nn.Module`。
- 关键难点：掌握 `view` 和 `transpose` 的张量维度变换。
- 维度检查：确保输入 `(Batch, Seq_len, Dim)` 进去，输出也是同样的形状。

### 3. 任务 C：可视化注意力权重
- 使用 seaborn 或 matplotlib 画出 Attention Matrix 的热力图。
- 实验：输入句子 "The animal didn't cross the street because it was too tired"，观察 "it" 是否对 "animal" 有更高的注意力。

---

## 第五阶段：复盘与预习

### 1. 今日闭环
- 你现在应该能解释：为什么 Attention 能实现并行计算？

### 2. 避坑指南
- 注意 $Q \cdot K^T$ 后的维度变化。
- 理解为什么 $V$ 决定了输出的含义，而 $Q, K$ 决定了分配的比例。

---

## 🛠 Day 2 工具与参考

- 在线阅读：Happy-LLM 第二章：注意力机制
- 可视化推荐：访问 Jay Alammar 的 Blog - The Illustrated Transformer。
- 课程重点：注意力机制的数学表示、为什么要有多头注意力机制、代码实现缩放点积注意力。

---

<a id="day3"></a>

# 📅 Day 3：Transformer 架构拆解（上）—— 结构细节

## 第一阶段：给模型注入"空间感"

**目标：解决 Attention 无法分辨语序的问题（位置编码）。**

### 1. 为什么需要位置编码 (Positional Encoding)?
- **核心痛点**：Attention 是并行的，对它来说"我吃鱼"和"鱼吃我"的计算结果完全一样。
- **解决方案**：在词向量（Embedding）上"加上"一个代表位置的向量。

### 2. 正弦余弦编码的数学美感 
- **公式推导**：理解不同频率的正弦和余弦波如何组合成唯一的坐标。
- **为什么用这种方式？**：允许模型学习到"相对位置"关系（因为 $\sin(a+b)$ 可以由 $\sin(a)$ 和 $\cos(b)$ 的线性组合表示）。

### 3. 阅读资料
- 阅读 Happy-LLM 第 2 章：位置编码部分。
- 思考：为什么是"相加"而不是"拼接"？

---

## 第二阶段：特征的二次加工

**目标：掌握前馈神经网络 (FFN)，理解它如何处理 Attention 提取的信息。**

### 1. 逐位置前馈网络 (Point-wise FFN)
- **结构**：两个线性层（Linear）中间夹一个激活函数（ReLU 或 GeLU）。
- **为什么叫"逐位置"？**：同一个句子里，每个单词用的都是同一套 FFN 参数，且单词之间互不干扰。
- **升维与降维**：通常将维度扩大 4 倍（如 512 -> 2048）再缩回去，目的是在更高维的空间进行非线性映射。

### 2. Checkpoint
- 对比：Attention 负责"收集信息"，FFN 负责"消化信息"。

---

## 第三阶段：训练的"保命符"——残差与归一化

**目标：理解为什么 Transformer 能堆叠几十层甚至上百层而不崩塌。**

### 1. 残差连接 (Residual Connection/Skip Connection) 
- **公式**：$$Output = Layer(x) + x$$
- **原理**：让梯度可以像坐电梯一样直接回到浅层，解决深度网络中的"梯度消失"问题。

### 2. 层归一化 (Layer Normalization)
- **对比 BN (Batch Norm)**：为什么 NLP 不用 BN？（因为序列长度不固定，Batch 统计不稳定）。
- **LN 的作用**：将每一层神经元的输出归一化为均值为 0、方差为 1 的分布，加速收敛。

### 3. Post-LN vs Pre-LN
- **重点**：LLaMA2 使用的是 Pre-LN（在 Attention 前做 Norm），这对稳定大规模训练至关重要。

---

## 第四阶段：手写代码实战

**目标：用 PyTorch 实现今天学到的三个核心模块。**

### 1. 任务 A：实现 PositionalEncoding 类 
- 使用 `torch.sin` 和 `torch.cos` 构建编码矩阵。
- 可视化：画出位置编码的热力图，观察斑马线一样的纹路。

### 2. 任务 B：实现 FeedForward 类
- 使用 `nn.Sequential` 快速搭建：Linear -> ReLU -> Linear。

### 3. 任务 C：封装 AddNorm 模块
- 实现残差连接和 `nn.LayerNorm`。
- 维度实验：确保相加时 $Layer(x)$ 和 $x$ 的 Shape 完全一致。

---

## 第五阶段：复盘与架构预览 

### 1. 知识大串联
- 一个完整的 Transformer 层流程：Input -> PosEncoding -> Attention -> AddNorm -> FFN -> AddNorm。

### 2. 今日闭环
- 你现在应该能解释：如果不加位置编码，Transformer 会变成什么？

---

## 🛠 Day 3 学习辅助

- 在线阅读：Happy-LLM 第二章：位置编码与 FFN
- 互动实验：在 Jupyter 中尝试修改 FFN 的中间维度（从 4 倍改到 1 倍），看看参数量和运算速度的变化。
- 课程重点：位置编码、Transformer 整体架构设计、残差与归一化。

---

<a id="day4"></a>

# 📅 Day 4：Transformer 架构拆解（下）—— 整体组装

## 第一阶段：构建 Encoder 

**目标：将零件封装成层，并堆叠成编码器。**

### 1. Encoder Layer 的封装
- **结构复习**：Input -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm。
- **代码实现逻辑**：如何使用 `nn.ModuleList` 来管理多个相同的层。

### 2. 数据的流动
- **理解张量（Tensor）在 Encoder 中的形状变换**：始终保持 $(Batch\_Size, Seq\_Len, Model\_Dim)$。
- **核心思考**：为什么每一层的输出维度都一样？

---

## 第二阶段：攻克 Decoder

**目标：掌握 Decoder 的特殊结构，特别是"掩码"和"交叉注意力"。**

### 1. 带掩码的自注意力 (Masked Self-Attention)
- **为什么要掩码？**：在预测第 $n$ 个词时，不能提前看到第 $n+1$ 个词的信息（防止"作弊"）。
- **实现原理**：Look-ahead Mask（上三角矩阵）。将未来位置的得分设为 $-\infty$，经过 Softmax 后权重变为 0。

### 2. 交叉注意力 (Encoder-Decoder Attention) 
- **Q、K、V 的来源**：Query 来自 Decoder（我想找什么），Key 和 Value 来自 Encoder 的输出（我有这些背景信息）。
- **这是 Seq2Seq 模型的核心**：Decoder 根据已生成的词去询问 Encoder 原文的意思。

### 3. 阅读资料
- 阅读 Happy-LLM 第 2 章：Decoder 详解部分。

---

## 第三阶段：最后的总装与概率输出 

**目标：完成整个 Transformer 类的构建，并理解如何从向量变回文字。**

### 1. Transformer 类的组装
- 集成 Embedding + Positional Encoding + Encoder + Decoder。

### 2. 线性层与 Softmax 
- **投影层**：将 $Model\_Dim$ 映射到 $Vocab\_Size$（词表大小）。
- **Softmax**：将输出转化为每个词出现的概率。

### 3. 损失函数 (Loss Function)
- **学习交叉熵损失 (Cross-Entropy Loss)**：如何衡量模型预测的概率分布与真实词之间的差距。

---

## 第四阶段：组装与测试

**目标：运行你亲手写出的完整 Transformer 模型。**

### 1. 任务 A：编写 Transformer 类 
- **要求**：严格按照公式定义 `forward` 函数。
- **维度追踪（最重要）**：在每一行代码注释中写出 Tensor 的 Shape。例如：`# x: [batch, seq_len, d_model]`。

### 2. 任务 B：维度测试
- 构造随机张量作为输入：`src = torch.randint(0, vocab_size, (batch, seq_len))`。
- 执行 `output = model(src, tgt)`。
- 成功标准：输出 Shape 为 `(batch, target_len, vocab_size)` 且没有报错。

### 3. 任务 C：可视化 Mask
- 打印并画出 Look-ahead Mask 的矩阵图，确保它是下三角阵。

---

## 第五阶段：复盘与深度思考

### 1. 知识地图梳理
- 在草稿上画出 Transformer 的简易结构图。

### 2. 今日闭环
- **核心问题**：如果 Decoder 没有 Mask 会发生什么？

---

## 🔍 Day 4 避坑与调试指南

- **维度对齐**：90% 的报错来自 Linear 层和 Attention 层的维度不匹配。请反复检查 `d_model` 和 `n_heads` 是否能整除。
- **Device 意识**：确保模型参数和输入张量都在同一个设备上（CPU 或 GPU）。
- **课程重点**：Transformer 的结构、掩码和交叉注意力的作用、维度对齐。

---

<a id="day5"></a>

# 📅 Day 5：预训练语言模型 (PLM) 的范式演进

## 第一阶段：PLM 的“三国鼎立”

**目标：理解 Transformer 的不同零件如何组合成不同的经典模型。**

### 1. Encoder-only (以 BERT 为代表)
- 核心任务：掩码语言模型 (MLM) ——“完形填空”。
- 优势：双向理解能力强。
- 擅长领域：文本分类、命名实体识别 (NER)、情感分析。

### 2. Decoder-only (以 GPT 系列为代表)
- 核心任务：自回归语言模型 (Causal LM) ——“下一词预测”。
- 优势：天生适合生成任务，易于扩展规模。
- 擅长领域：文本生成、对话、创意写作。

### 3. Encoder-Decoder (以 T5, BART 为代表)
- 核心任务：Seq2Seq 转换。
- 优势：灵活，将所有 NLP 任务统一为“文本到文本”。
- 擅长领域：机器翻译、文本摘要。

---

## 第二阶段：深挖——为什么 Decoder-only 赢了？

**目标：理解大模型时代的架构共识。**

### 1. 统一的范式：为什么现在的 LLM 几乎全是 Decoder-only？
- 零样本/少样本能力 (Zero-shot/Few-shot)：GPT 证明了只要模型足够大，预测下一个词就能学会推理。
- 计算效率：在推断时，Decoder-only 的 KV Cache 机制非常高效。
- 训练规模：Decoder-only 架构在超大规模参数下表现最稳定。

### 2. 阅读资料
- 阅读 Happy-LLM 第 3 章：预训练语言模型演进。
- 思考题：为什么 BERT 虽然理解能力强，但在做对话机器人时不如 GPT 灵活？

---

## 第三阶段：拥抱生态——Hugging Face 实战

**目标：学会使用工业界标准的库，不再从零重复造轮子。**

### 1. Transformers 库核心组件
- `AutoTokenizer`：自动加载对应模型的分词器。
- `AutoModel` / `AutoModelForCausalLM`：自动加载模型架构和权重。

### 2. Pipeline 的魔力
- 学习用 3 行代码实现：文本分类、生成、翻译。

### 3. 【实战演练】
- 在 Colab 或本地环境中，分别加载 `bert-base-uncased` 和 `gpt2`。
- 对比任务：给 BERT 一个带 `[MASK]` 的句子，看它补全什么；给 GPT2 一个开头，看它往下写什么。

---

## 第四阶段：动手体验——生成式模型的威力

**目标：亲手写一个简单的解码策略。**

### 1. 解码策略 (Decoding Strategy) 学习
- 贪心搜索 (Greedy Search)：永远选概率最大的，容易陷入死循环。
- 束搜索 (Beam Search)：保留多个候选路径。
- 采样 (Sampling) 与 温度 (Temperature)：增加生成的多样性和创造性。

### 2. 【代码任务】
- 使用 GPT-2 模型手动编写一个循环，每次预测一个词，并拼接到输入中，实现简易版的“打字机”生成效果。
- 尝试调整 `temperature` 参数（0.1 vs 1.0），观察生成内容的逻辑性变化。

---

## 第五阶段：第一阶段复盘与晋级预告

### 1. 五天回顾
- Day 1: 文本变向量（Embedding）。
- Day 2: 向量交互（Attention）。
- Day 3: 架构细节（Positional, FFN, Norm）。
- Day 4: 总装（Encoder/Decoder）。
- Day 5: 家族演化（BERT/GPT/T5）。

### 2. 知识通关自测
- 你能解释从 Word2Vec 到 GPT-4，模型处理上下文信息的能力发生了什么本质变化吗？

### 3. 明日开启：第二阶段——深挖原理
- 我们将进入“大模型时代”，探讨 Scaling Law（规模法则）和 LLM 的涌现能力。

---

## 🛠 Day 5 学习工具

- 官方文档：Hugging Face Transformers Quickstart
- 在线体验：Hugging Face Spaces（去玩一玩别人部署好的各种模型）。
- 课程重点：为什么选择 Decoder-only、Transformers 库实战、解码策略。