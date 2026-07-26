import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from warnings import warn

from nets.resnet import resnet18, resnet50, resnet101, resnet152
from nets.efficientNet import EfficientNet

from attention.SelfGatedDiGroupAttention.SelfGatedDiGroupAttention import SelfGatedDiGroupAttention as Attention
from attention.ECA import ECA_layer

import torch
from torch import nn

from utils.CARAFE import CARAFE  
from nets.eutils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    [修改]: 增加 use_eca 参数，支持将 SE 替换为 ECA
    """

    def __init__(self, block_args, global_params, image_size=None, use_eca=True):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        
        self.use_eca = use_eca
        
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1) and (not self.use_eca)
        
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        if self.use_eca:
            self.eca_layer = ECA_layer(oup, k_size=3)         
        elif self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.use_eca:
            x = self.eca_layer(x)
        elif self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


'''
使用PSPNet的PPM模块
PPM中的2,4,6池化修改为位移池化
新增：完整支持EfficientNet-b5预训练模型
'''
class ShiftPooling(nn.Module):
    """
    优化后的位移平均池化：
    1. 反射填充 (Reflect Padding)：避免边界死区。
    2. 平均融合 (Mean Fusion)：更平滑的特征过渡，符合 PPM 逻辑。
    3. 显存优化：摒弃 torch.stack，采用累加求平均。
    """
    def __init__(self, kernel_size: int, stride: int, shift_offset: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.shift_offset = shift_offset
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        
        self.shift_right = None
        self.shift_down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        offset = self.shift_offset

        self.shift_right = F.pad(x, (offset, 0, 0, 0), mode='reflect')[:, :, :, :W]
        self.shift_down = F.pad(x, (0, 0, offset, 0), mode='reflect')[:, :, :H, :]

        x_pool = self.pool(x)
        sr_pool = self.pool(self.shift_right)
        sd_pool = self.pool(self.shift_down)

        merged_x = (x_pool + sr_pool + sd_pool) / 3.0
        
        return merged_x

class PyramidPoolingModule(nn.Module):
    """
    优化版 PPM：
    1. 统一归一化：全线使用 BatchNorm2d。
    2. 残差连接：支持通道不匹配时的自动映射。
    """
    def __init__(
        self, 
        in_channels: int = 128, 
        out_channels: int = 256, 
        pool_scales: List[int] = [1, 2, 3, 6],
        reduce_ratio: int = 4
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.reduce_dim = max(in_channels // reduce_ratio, 1)
        
        self.pool_branches = nn.ModuleList()
        for scale in pool_scales:
            branch = []
            if scale == 1:
                branch.append(nn.AdaptiveAvgPool2d(scale))
                branch.extend([
                    nn.Conv2d(in_channels, self.reduce_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.reduce_dim),                     nn.ReLU(inplace=True)
                ])
            else:
                branch.append(ShiftPooling(kernel_size=scale, stride=scale))
                branch.extend([
                    nn.Conv2d(in_channels, self.reduce_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.reduce_dim),                     nn.ReLU(inplace=True)
                ])
            self.pool_branches.append(nn.Sequential(*branch))
        
        concat_channels = in_channels + self.reduce_dim * len(pool_scales)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)             )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        pool_outs = [x]         
        h, w = x.shape[2:]
        for branch in self.pool_branches:
            out = branch(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pool_outs.append(out)
        
        feat = torch.cat(pool_outs, dim=1)
        out = self.fusion_conv(feat)
        
        out = out + self.shortcut(identity)
        return out
class UpsampleModule(nn.Module):
    """
    上采样辅助模块：支持 Bilinear 和 CARAFE
    """
    def __init__(self, in_channels: int, out_channels: int, mode: str = 'carafe'):
        super().__init__()
        self.mode = mode
        
        if mode == 'carafe':
            self.channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            c_mid = max(32, out_channels // 4)
            self.carafe = CARAFE(c=out_channels, c_mid=c_mid, scale=2, k_enc=3, k_up=5)
            
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'carafe':
            x = self.channel_adjust(x)
            x = self.carafe(x)                 
        else:
            x = self.up(x)
            x = self.conv1x1(x)
            x = self.bn(x)
        return x
    
import collections
MockBlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'input_filters', 'output_filters', 
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'
])

MockGlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon'
])

class UnetUpBlock(nn.Module):
    """
    Unet上采样拼接模块：原生集成 EfficientNet 官方 MBConvBlock
    [修复]: 增加注意力机制的通道适配器，解决 skip 和 up 通道不一致导致的 crash
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, up_channels: int, attention_module: Optional[nn.Module] = None):
        super().__init__()
        self.attention = attention_module
        
        self.att_gate_adapter = None
        if self.attention is not None and up_channels != skip_channels:
            self.att_gate_adapter = nn.Sequential(
                nn.Conv2d(up_channels, skip_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(skip_channels),
                nn.ReLU(inplace=True)
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True) 
        )
        
        block_args = MockBlockArgs(
            kernel_size=3,
            input_filters=out_channels,
            output_filters=out_channels,
            expand_ratio=4,                 id_skip=True,                   stride=1,
            se_ratio=None               )
        
        global_params = MockGlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3
        )
        
        self.mbconv = MBConvBlock(block_args=block_args, global_params=global_params, image_size=None)
        self.final_act = nn.SiLU(inplace=True)
    
    def forward(self, skip_feat: torch.Tensor, up_feat: torch.Tensor) -> torch.Tensor:
        if skip_feat.shape[2:] != up_feat.shape[2:]:
            up_feat_aligned = F.interpolate(
                up_feat, size=skip_feat.shape[2:], 
                mode='bilinear', align_corners=False
            )
        else:
            up_feat_aligned = up_feat
        
        skip_feat_processed = skip_feat
        if self.attention:
            gate_feat = up_feat_aligned
            if self.att_gate_adapter is not None:
                gate_feat = self.att_gate_adapter(up_feat_aligned)
            
            skip_feat_processed = self.attention(skip_feat, gate_feat)

        concat_feat = torch.cat([skip_feat_processed, up_feat_aligned], dim=1)
        
        x = self.fusion_conv(concat_feat)
        
        x = self.mbconv(x, drop_connect_rate=None)
        
        x = self.final_act(x)

        return x

class Unet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = False,
        backbone: str = 'efficientnet-b5',
        is_print_size: bool = False,
        use_attention   : bool = 1,
        use_psp         : bool = 1,
        psp_scales: List[int] = [1,2,3,6],
        psp_reduce_ratio: int = 4,
        device: Optional[torch.device] = None,
        upsample_mode   : str = 'bilinear',
        decoder_channels: Optional[List[int]] = [320, 160, 80, 48, 24],         **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_name = backbone
        self.is_print_size = is_print_size
        self.use_attention = use_attention
        self.use_psp = use_psp
        self.upsample_mode = upsample_mode
        self.psp_scales = psp_scales
        self.psp_reduce_ratio = psp_reduce_ratio
        
        self.decoder_channels_config = decoder_channels
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = self._init_backbone()
        
        if self.backbone_name.startswith('efficientnet'):
            if hasattr(self.backbone, '_fc'): del self.backbone._fc
            if hasattr(self.backbone, '_avg_pooling'): del self.backbone._avg_pooling
            if hasattr(self.backbone, '_dropout'): del self.backbone._dropout
        
        self.backbone.to(self.device)

        self._build_decoder()

    def _init_backbone(self) -> nn.Module:
        """封装主干网络初始化逻辑：支持ResNet + EfficientNet（含b5预训练）"""
        if self.backbone_name.startswith('resnet'):
            backbone_map = {
                'resnet18': resnet18,
                'resnet50': resnet50,
                'resnet101': resnet101,
                'resnet152': resnet152
            }
            if self.backbone_name not in backbone_map:
                raise NotImplementedError(f"不支持的ResNet变体: {self.backbone_name}")
            backbone = backbone_map[self.backbone_name](pretrained=self.pretrained)
        
        elif self.backbone_name.startswith('efficientnet'):
            try:
                if self.pretrained:
                    backbone = EfficientNet.from_pretrained(
                        self.backbone_name,
                        advprop=True,                          in_channels=3,
                        num_classes=1000                      )
                    print(f"✅ 成功加载{self.backbone_name}预训练权重")
                else:
                    backbone = EfficientNet.from_name(self.backbone_name)
                    print(f"ℹ️ 未加载{self.backbone_name}预训练权重，使用随机初始化")
            except Exception as e:
                warn(f"⚠️ 加载{self.backbone_name}预训练权重失败: {e}，使用随机初始化")
                backbone = EfficientNet.from_name(self.backbone_name)
        
        else:
            raise NotImplementedError(f"不支持的主干网络: {self.backbone_name}")
        
        return backbone

    def _get_encoder_features(self, dummy_input: torch.Tensor) -> List[torch.Tensor]:
        """提取编码器特征：区分ResNet/EfficientNet的特征格式"""
        if self.backbone_name.startswith('efficientnet'):
            endpoints = self.backbone.extract_endpoints(dummy_input)
            features = [endpoints[f'reduction_{i}'] for i in range(1, 6)]
        
        else:
            features = self.backbone(dummy_input)
        
        if len(features) < 4:
            warn(f"编码器特征数量不足（{len(features)}），可能导致解码器构建失败")
        
        return features

    def _build_decoder(self) -> None:
        """动态构建解码器：支持自定义通道数"""
        print(f"\n--- 开始为{self.backbone_name}构建解码器 ---")
        
        input_size = self._get_input_size()
        dummy_input = torch.randn(2, 3, input_size, input_size).to(self.device)

        with torch.no_grad():
            self.backbone.eval()
            features = self._get_encoder_features(dummy_input)
            encoder_channels = [feat.shape[1] for feat in features]
            print(f"自动检测编码器通道数: {encoder_channels}")

            
            if self.decoder_channels_config is None:
                target_channels = [encoder_channels[i] for i in range(len(features)-2, -1, -1)]
                target_channels.append(target_channels[-1] // 2)
            else:
                if len(self.decoder_channels_config) < 5:
                    warn(f"提供的 decoder_channels 长度不足 ({len(self.decoder_channels_config)}), 期望 5。将尝试自动补全。")
                target_channels = self.decoder_channels_config
            
            print(f"解码器目标通道数配置: {target_channels}")

            self.psp_module = None
            if self.use_psp:
                bottleneck_channels = encoder_channels[-1]
                self.psp_module = PyramidPoolingModule(
                    in_channels=bottleneck_channels,
                    out_channels=bottleneck_channels,
                    pool_scales=self.psp_scales,
                    reduce_ratio=self.psp_reduce_ratio
                ).to(self.device)

            self.decoder_stages = nn.ModuleList()
            current_channels = encoder_channels[-1] 
            bottleneck_near_skip_indices = [len(features)-2, len(features)-3, len(features)-4]

            for idx, i in enumerate(range(len(features)-2, -1, -1)):
                skip_channels = encoder_channels[i]                     
                out_channels = target_channels[idx]                     
                print(f"阶段 {idx+1}: Input({current_channels}) + Skip({skip_channels}) -> Output({out_channels})")

                up_module = UpsampleModule(
                    in_channels=current_channels, 
                    out_channels=out_channels,                     mode=self.upsample_mode
                )
                
                attention_module = None
                if self.use_attention:
                    if i not in bottleneck_near_skip_indices:
                        attention_module = Attention(skip_channels)

                up_concat_module = UnetUpBlock(
                    in_channels=skip_channels + out_channels,                     out_channels=out_channels,                                    skip_channels=skip_channels,                                  up_channels=out_channels,                                     attention_module=attention_module
                )

                self.decoder_stages.append(nn.ModuleDict({
                    'up': up_module,
                    'up_concat': up_concat_module
                }))

                current_channels = out_channels 
            final_out_channels = target_channels[4] if len(target_channels) > 4 else current_channels // 2
            
            print(f"最终阶段: Input({current_channels}) -> Output({final_out_channels}) -> Class({self.num_classes})")

            self.final_up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(current_channels, final_out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(final_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_out_channels, final_out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(final_out_channels),
                nn.ReLU(inplace=True),
            )
            self.final_conv = nn.Conv2d(final_out_channels, self.num_classes, kernel_size=1)

        self.backbone.train()
        self.decoder_stages.to(self.device)
        self.final_up_conv.to(self.device)
        self.final_conv.to(self.device)
        print("--- 解码器构建完成 ---\n")

    def _get_input_size(self) -> int:
        """获取主干网络对应的默认输入尺寸：适配EfficientNet-b5的456"""
        if self.backbone_name.startswith('efficientnet'):
            return EfficientNet.get_image_size(self.backbone_name)
        return 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：兼容ResNet/EfficientNet + PPM"""
        x = x.to(self.device)
        
        features = self._get_encoder_features(x)
        
        if self.is_print_size:
            print("\n---- 编码器特征图尺寸 ----")
            for idx, feat in enumerate(features):
                print(f"特征层{idx+1}: {feat.shape}")
        
        x = features[-1]
        if self.use_psp and self.psp_module is not None:
            x = self.psp_module(x)
            if self.is_print_size:
                print(f"PPM处理后瓶颈特征尺寸: {x.shape}")
        
        for stage_idx, stage in enumerate(self.decoder_stages):
            skip_idx = len(features) - 2 - stage_idx
            skip_feat = features[skip_idx]
            
            x = stage['up'](x)
            x = stage['up_concat'](skip_feat, x)
            
            if self.is_print_size:
                print(f"解码器阶段{stage_idx+1}输出尺寸: {x.shape}")
        
        x = self.final_up_conv(x)
        if self.is_print_size:
            print(f"最终上采样输出尺寸: {x.shape}")
        
        output = self.final_conv(x)
        return output

    def freeze_backbone(self, freeze_all: bool = True, freeze_layers: Optional[List[str]] = None) -> None:
        """冻结主干网络参数：兼容ResNet/EfficientNet"""
        print("冻结主干网络参数...")
        for name, param in self.backbone.named_parameters():
            if freeze_all:
                param.requires_grad = False
            else:
                if freeze_layers and any(layer in name for layer in freeze_layers):
                    param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """解冻主干网络所有参数"""
        print("解冻主干网络参数...")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Tuple[int, int]:
        """统计模型参数：可训练/总参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params

if __name__ == "__main__":
    TEST_BACKBONES = ['resnet50','efficientnet-b5']
    NUM_CLASSES = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {DEVICE}")

    for backbone in TEST_BACKBONES:
        print(f"\n=========== 测试 {backbone} 主干网络（集成PPM） ===========")
        
        model = Unet(
            num_classes=NUM_CLASSES,
            backbone=backbone,
            pretrained=True if backbone == 'efficientnet-b6' else False,
            is_print_size=True,
            use_attention=True,
            use_psp=True,
            psp_scales=[1,2,3,6],
            psp_reduce_ratio=4,
            device=DEVICE
        )
        model.eval()

        input_size = [320,512]
        input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        expected_shape = (1, NUM_CLASSES, input_size[0], input_size[1]) 
        is_match = output.shape == expected_shape
        print(f"\n输入尺寸: {input_tensor.shape}")
        print(f"输出尺寸: {output.shape} | 期望尺寸: {expected_shape} -> {'匹配' if is_match else '不匹配'}")
        
        try:
            from thop import profile
            flops, params = profile(model, inputs=(input_tensor,))
            print(f"模型总参数: {params/1e6:.2f}M")
            print(f"模型FLOPs: {flops/1e9:.2f}G")
        except ImportError:
            print("未安装thop，跳过参数量计算")
        except Exception as e:
            print(f"计算参数量失败: {e}")