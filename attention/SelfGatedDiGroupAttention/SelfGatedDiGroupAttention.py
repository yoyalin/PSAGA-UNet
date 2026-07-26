import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfGatingLayer(nn.Module):
    """
    自门控层（Self-Gating Layer）
    """
    def __init__(self, in_channels):
        super(SelfGatingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.conv(x))  
        return x * gate  


class SelfGatedDiGroupAttention(nn.Module):
    """
    自门控双输入分组注意力机制（Self-Gated Dual Input Grouped Attention）
    + [新增特性] Channel Shuffle (通道打乱)：促进组间信息交互
    """
    def __init__(self, channels, channels2=None, k_size=3, factor=16):
        super(SelfGatedDiGroupAttention, self).__init__()
        
        self.channels1 = channels
        self.channels2 = channels2 if channels2 is not None else channels
        
        if self.channels1 != self.channels2:
            self.align_conv = nn.Sequential(
                nn.Conv2d(self.channels2, self.channels1, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels1)
            )
        else:
            self.align_conv = nn.Identity()

        self.groups = factor
        assert self.channels1 >= 1, f"主通道数{self.channels1}需≥1"
        
        self.group_channels = (self.channels1 + self.groups - 1) // self.groups
        self.target_channels = self.groups * self.group_channels 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv2d(1, 1, kernel_size=(k_size, 1), 
                                  padding=((k_size-1)//2, 0), bias=False)
        self.ca_fusion_conv = nn.Conv2d(2 * self.group_channels, 
                                        self.group_channels, 
                                        kernel_size=1, stride=1, padding=0)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.sa_fusion_conv = nn.Conv2d(self.group_channels, 
                                        self.group_channels, 
                                        kernel_size=1, stride=1, padding=0)
        
        self.self_gating = SelfGatingLayer(self.group_channels)

    def _pad_channels(self, x):
        """通道维度补零"""
        current_c = x.shape[1]
        pad_channels = self.target_channels - current_c
        if pad_channels > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_channels))
        return x, pad_channels

    def channel_shuffle(self, x, groups):
        """
        [新增] 通道打乱操作
        x: [B, C, H, W]
        groups: int
        """
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x1, x2):
        x2_aligned = self.align_conv(x2)
        
        assert x1.shape[2:] == x2_aligned.shape[2:], "空间尺寸不匹配"
        b, c, h, w = x1.size()
        
        x1_padded, pad1 = self._pad_channels(x1)
        x2_padded, pad2 = self._pad_channels(x2_aligned)
        
        group_x1 = x1_padded.reshape(b * self.groups, self.group_channels, h, w)
        group_x2 = x2_padded.reshape(b * self.groups, self.group_channels, h, w)
        
        y1 = self.avg_pool(group_x1).permute(0, 2, 1, 3)
        y2 = self.avg_pool(group_x2).permute(0, 2, 1, 3)
        
        y1 = self.conv1d(y1).permute(0, 2, 1, 3)
        y2 = self.conv1d(y2).permute(0, 2, 1, 3)
        
        ca_combined = torch.cat((y1, y2), dim=1)
        ca_combined = self.ca_fusion_conv(ca_combined)
        ca_gate = self.self_gating(ca_combined)
        
        ca_output = group_x1 * ca_gate.broadcast_to(group_x1.shape)
        
        x_h = self.pool_h(ca_output)
        x_w = self.pool_w(ca_output).permute(0, 1, 3, 2)
        
        sa_combined = torch.cat([x_h, x_w], dim=2)
        sa_combined = self.sa_fusion_conv(sa_combined)
        
        x_h, x_w = torch.split(sa_combined, [h, w], dim=2)
        sa_x_h = self.self_gating(x_h)
        sa_x_w = self.self_gating(x_w).permute(0, 1, 3, 2)
        
        sa_x_h = sa_x_h.reshape(b * self.groups, self.group_channels, h, 1)
        sa_x_w = sa_x_w.reshape(b * self.groups, self.group_channels, 1, w)
        sa_output = (sa_x_h * sa_x_w).reshape(b, self.target_channels, h, w)
        
        out_padded = x1_padded * x2_padded * sa_output
        
        out_shuffled = self.channel_shuffle(out_padded, self.groups)
        
        if pad1 > 0:
            outputs = out_shuffled[:, :-pad1, :, :]
        else:
            outputs = out_shuffled
        
        return outputs

if __name__ == "__main__":
    torch.manual_seed(42)
    x1 = torch.randn(2, 62, 32, 32)
    x2 = torch.randn(2, 62, 32, 32)
    
    attention = SelfGatedDiGroupAttention(channels=62, factor=4)
    out = attention(x1, x2)
    
    print(f"输入形状: {x1.shape}")
    print(f"输出形状: {out.shape}")
    print("Channel Shuffle 集成测试通过。")