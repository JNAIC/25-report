# 第六周周报
## 学习内容
1. 继续学习LLM比赛解决‘
2. 观看李沐视频

## 具体内容
### 一、LLM代码
#### 配置
1. **deberta_v3_extra_small_en模型**：
    DeBERTa是Transformer的一种变体，它改变了自注意力机制：
     - 使用使用“disentangled attention”，把词语内容和位置编码分开处理。
     - 能够更好的捕捉词与词之间的相对位置关系。
2.  **学习率调度器**
      - 学习率：
      学习率决定了模型参数每次更新的脚步大小，步伐太大或则过小都会影响训练效果
      - 为什么需要调度学习率
      因为模型参数的训练并不是每一步都适合同一个学习率。像初期更适合采用较大的学习率来快速找到大致方向，而在后期适合采用较小的学习率来进行精细的调整。
      - 学习率调度器的作用
      1. 保持稳定训练
      2. 避免跳过较好的解
      3. 提高最终精度
       - 常见的调度器类型
      4. 固定学习率（ConstantLR）
      5. 阶梯式学习率（StepLR）
      6. 余弦退火学习率（CosineAnnealingLR）
      7. 指数衰减学习率（ExponentialLR）
      8. 循环学习率（CyclicLR）
      9. 自适应学习率（如ReduceLROnPlateau）
       - 常用的调度器介绍
       1. Step LR
            - 原理：每隔一定的训练步骤或epoch数，学习率就会乘以一个预设的衰减因子，使得其按比例降低学习率。
            - 公式：$$
    \eta_t = \eta_0 \cdot \gamma^{\lfloor \frac{t}{\text{step\_size}} \rfloor}
    $$ 其中，$\eta_t$ 是第 $t$ 步的学习率，$\eta_0$ 是初始学习率，$\gamma$ 是衰减因子，$\text{step\_size}$ 是每隔多少步进行一次衰减。
            - 特点：
      学习率会突然下降，训练过程中会出现"跳跃式"收敛。由此导致其比较简单有效，但是步长和衰减因子需要调节。常用于CNN，ResNet等图像任务中。
       2. Cosine Annealing LR
            - 原理：学习率按照余弦函数的形式逐渐降低，从初始值逐渐减小到一个最小值，然后再重新开始一个新的周期。
            - 公式：$$
    \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos \frac{t \pi}{T}\right)
    $$其中，$\eta_t$ 是第 $t$ 步的学习率，$\eta_{\max}$ 是初始学习率，$\eta_{\min}$ 是最小学习率，$T$ 是一个周期的长度。
            - 特点：
      学习率会平滑地降低，训练过程中会出现"平滑式"收敛。由此导致其比较适合于需要细致调整的任务，如NLP中的Transformer模型等。
      10. Exponential LR
            - 原理：学习率按照指数函数的形式逐渐降低，每一步的学习率都是前一步的学习率乘以一个固定的衰减因子。
            - 公式：$$
    \eta_t = \eta_0 \cdot \gamma^t
    $$ 其中，$\eta_t$ 是第 $t$ 步的学习率，$\eta_0$ 是初始学习率，$\gamma$ 是衰减因子。
            - 特点：
      学习率会以指数形式快速降低，适用于需要快速收敛的任务。

3. **字典推导式**
字典推导式（dict comprehension）是 Python 里一种用一行代码快速构造字典的写法，和列表推导式类似，但生成的是 {key: value} 结构。
- 基本格式
```python 
 {key表达式: value表达式 for 变量 in 可迭代对象 if 条件}
``` 
- 输出的结果
```python
{key1:value1,key2:value2,......}
```

```python 
class CFG: #定义配置类CFG
    seed = 42  # 设置随机种子，设置为42可确保复现时为同一个值，实际上可以任取
    preset ="deberta_v3_extra_small_en" # present 指定所使用的预训练模型名称或模型配置标识
    # deberta_v3_extra_small_en表示指定使用预训练模型为 DEBERTa v3模型的极小型英文版本 
    sequence_length = 512  # 表示输入文本的最大长度（token数），模型会将文本截断或者填充为512这个长度
    epochs = 3 # 训练次数
    batch_size = 16  # 表示每次训练使用的样本数量
    scheduler = 'cosine'  # scheduler指定学习率调度器类型
                        #cosine表示采用余弦退火学习率调度，学习率会按照余弦曲线线性变化，从初始值逐渐减小 
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2:'winner_tie'} # 设置一个从标签编号-->类别名称的映射表。
    # 在此处‘0’代表 "winner_model_a"   ‘1’ 代表 "winner_model_b"  ‘2’ 代表 "winner_tie"
    name2label = {v:k for k, v in label2name.items()} # 通过字典推导式构建类别名称-->标签编号的映射表
    class_labels = list(label2name.keys()) # 将label2name的所有key取出，变成列表
    class_names = list(label2name.values())
```
