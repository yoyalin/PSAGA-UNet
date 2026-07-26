import torch
from torch import nn


class ECA_layer(nn.Module):
    """构建一个 ECA 模块。

    参数:
        channel: 输入特征图的通道数
        k_size: 自适应选择的一维卷积核大小
    """
    def __init__(self, channel, k_size=3,factor = None):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数，定义数据流经过该模块的处理步骤。

        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
        Tensor: 经过ECA模块处理后的输出张量。
        """
        
        y = self.avg_pool(x)
        
        y = y.squeeze(-1).transpose(-1, -2)
        
        y = self.conv(y)
        
        y = y.transpose(-1, -2)
        
        y = y.unsqueeze(-1)
        
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)
    eca = ECA_layer(channel=64)
    y = eca(x)
    print(y.shape)  