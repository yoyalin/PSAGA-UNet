import torch
from torch import nn
from ultralytics.nn.modules import Conv


class CARAFE(nn.Module):
    """
    CARAFE 是一种上采样模块，通过学习的权重对特征图进行上采样。
    参数:
        c (int): 输入通道数
        k_enc (int): 编码器部分的卷积核大小
        k_up (int): 上采样时使用的 unfold 核大小
        c_mid (int): 中间通道数
        scale (int): 上采样倍率
    """

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super(CARAFE, self).__init__()
        self.scale = scale  
        self.comp = Conv(c, c_mid)

        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)

        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')

        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()          
        h_, w_ = h * self.scale, w * self.scale  
        W = self.comp(X)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)  
        X = self.upsmp(X)
        X = self.unfold(X)
        X = X.view(b, c, -1, h_, w_)  
        X = torch.einsum('bkhw,bckhw->bchw', [W, X])          
        return X  