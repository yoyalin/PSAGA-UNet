import torch.nn as nn

class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels = None, kernel_size=3, padding=1, bias=False, dilation=1, **kwargs):
        super(DWConvLayer, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,dilation=dilation,
                                   padding=padding, groups=in_channels, bias=bias, **kwargs)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x