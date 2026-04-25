# 第六周周报
## 学习内容
1. 继续学习LLM比赛解决‘
2. 观看李沐视频

## 具体内容
### 一、LLM代码
#### 配置
- deberta_v3_extra_small_en模型：
DeBERTa是Transformer的一种变体，它改变了自注意力机制：
  - 使用使用“disentangled attention”，把词语内容和位置编码分开处理。
  - 能够更好的捕捉词与词之间的相对位置关系。
- 学习率调度器
  - 学习率：
  学习率决定了模型参数每次更新的脚步大小，步伐太大或则过小都会影响训练效果
  - 为什么需要调度学习率
  因为模型参数的训练并不是每一步都适合同一个学习率。像初期更适合采用较大的学习率来快速找到大致方向，而在后期适合采用较小的学习率来进行精细的调整。
  - 学习率调度器的作用
  1. 保持稳定训练
  2. 避免跳过较好的解
  3. 提高最终精度
   - 常见的调度器类型
  1. 固定学习率（ConstantLR）
  2. 阶梯式学习率（StepLR）
  3. 余弦退火学习率（CosineAnnealingLR）
  4. 指数衰减学习率（ExponentialLR）
  5. 循环学习率（CyclicLR）
  6. 自适应学习率（如ReduceLROnPlateau）
````python 
class CFG: #定义配置类CFG
    seed = 42  # 设置随机种子，设置为42可确保复现时为同一个值，实际上可以任取
    preset ="deberta_v3_extra_small_en" # present 指定所使用的预训练模型名称或模型配置标识
    # deberta_v3_extra_small_en表示指定使用预训练模型为 DEBERTa v3模型的极小型英文版本 
    sequence_length = 512  # 表示输入文本的最大长度（token数），模型会将文本截断或者填充为512这个长度
    epochs = 3 # 训练次数
    batch_size = 16  # 表示每次训练使用的样本数量
    scheduler = 'cosine'  # scheduler指定学习率调度器类型
                        #cosine表示采用余弦退火学习率调度，学习率会按照余弦曲线线性变化，从初始值逐渐减小 
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())
