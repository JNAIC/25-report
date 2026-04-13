# 第三周周报
## 本周学习内容
- CNN的设计理念
- CNN各模块的功能与定义
- 运用AI编写对应代码块


---
以下为代码
```pytorch
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        # 输入假设: 1通道(灰度图), 尺寸 28x28。即 shape=[batch_size, 1, 28, 28]

        # 1. 第一层：卷积层 (对应图中的 C1)
        # 参数：输入通道1，输出通道6，卷积核5x5，填充2，默认步幅1
        # 尺寸计算：(28 - 5 + 2*2) / 1 + 1 = 28
        # 输出特征图：6 @ 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        
        # 2. 第二层：平均汇聚层/池化层 (对应图中的 S2)
        # 参数：池化窗口2x2，步幅2
        # 尺寸计算：(28 - 2) / 2 + 1 = 14
        # 输出特征图：6 @ 14x14
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 3. 第三层：卷积层 (对应图中的 C3)
        # 参数：输入通道6，输出通道16，卷积核5x5，无填充(即padding=0)
        # 尺寸计算：(14 - 5 + 0) / 1 + 1 = 10
        # 输出特征图：16 @ 10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 4. 第四层：平均汇聚层/池化层 (对应图中的 S4)
        # 参数：池化窗口2x2，步幅2
        # 尺寸计算：(10 - 2) / 2 + 1 = 5
        # 输出特征图：16 @ 5x5
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 5. 第五层：全连接层 (对应图中的 120-F5)
        # 注意：在进入全连接层前，需要把 16个 5x5 的二维特征图“展平”成一维向量
        # 输入神经元个数 = 通道数 * 高 * 宽 = 16 * 5 * 5 = 400
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        
        # 6. 第六层：全连接层 (对应图中的 84-F6)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        # 7. 第七层：输出层 (对应图中的 全连接层(10))
        # 输出10个特征，代表数字 0-9 的预测概率
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
        # 激活函数：原版 LeNet 常用 Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 追踪前向传播的维度变化
        x = self.conv1(x)         # [batch_size, 6, 28, 28]
        x = self.sigmoid(x)       # 激活函数不改变尺寸
        x = self.avgpool1(x)      # [batch_size, 6, 14, 14]
        
        x = self.conv2(x)         # [batch_size, 16, 10, 10]
        x = self.sigmoid(x)
        x = self.avgpool2(x)      # [batch_size, 16, 5, 5]
        
        # 展平操作 (Flatten)：将四维张量变为二维张量 [batch_size, 400]
        x = torch.flatten(x, 1)   
        
        x = self.fc1(x)           # [batch_size, 120]
        x = self.sigmoid(x)
        
        x = self.fc2(x)           # [batch_size, 84]
        x = self.sigmoid(x)
        
        x = self.fc3(x)           # [batch_size, 10]
        return x

# --- 测试代码 ---
if __name__ == "__main__":
    # 实例化网络
    net = LeNet()
    
    # 模拟生成一张 28x28 的单通道黑白图片 (Batch Size 设为 1)
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    
    # 传入网络进行计算
    output = net(X)
    
    print(f"输入尺寸: {X.shape}")
    print(f"输出尺寸: {output.shape}") 
    # 预期打印结果应为：torch.Size([1, 10])
```