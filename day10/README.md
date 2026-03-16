### 🚀 使用方法
1. 安装依赖

```
pip install -r requirements.txt
```
2. 准备数据

- 将 TinyStories 数据放入 ./data 目录
- 确保 tokenizer 文件位于 ./tokenizer.model
3. 启动训练

```
# 使用真实数据
python train.py --data_path ./data 
--tokenizer_path ./tokenizer.model

# 使用合成数据测试（无需真实数据）
python train.py --synthetic

# 使用 WandB 监控
python train.py --synthetic --use_wandb
```
4. 文本生成

```
python generate.py --model_path ./checkpoints/
checkpoint_500.pth --prompt "Once upon a time"
```
### 🔧 核心特性
功能 实现细节 优化器 AdamW with weight decay 学习率调度 Cosine Annealing with Warmup 混合精度 torch.cuda.amp (FP16) 梯度裁剪 grad_clip=1.0 采样策略 Temperature, Top-K, Top-P 监控 TensorBoard + WandB

### 📈 训练监控
- TensorBoard : tensorboard --logdir ./logs
- 关键指标 : 观察 train/loss 是否从 ~10 下降到 4-5
### 💡 炼丹师提示
1. 显存不足 ：减小 batch_size (32→16) 或 max_seq_len (256→128)
2. Loss 不下降 ：检查学习率是否合适，尝试调整 warmup_iters
3. 输出乱码 ：模型需要更多训练步数，耐心等待
