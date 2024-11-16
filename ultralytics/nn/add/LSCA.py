
from typing import Optional
import torch.nn as nn
import torch

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # Add more normalization types if needed
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # Add more activation types if needed
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class LSK1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 定义各个卷积层
        # 深度可分离卷积，保持输入和输出通道数一致，卷积核大小为5
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间卷积，卷积核大小为7，膨胀率为3，增加感受野
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 1x1卷积，用于降维
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 结合平均和最大注意力的卷积
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后的1x1卷积，将通道数恢复到原始维度
        self.conv = nn.Conv2d(dim // 2, dim, 1)



    def forward(self, x):


        attn1 = self.conv0(x)  # 第一个卷积特征

        attn2 = self.conv_spatial(attn1)  # 空间卷积特征

        # 对卷积特征进行1x1卷积以降维
        attn1 = self.conv1(attn1)

        attn2 = self.conv2(attn2)

        # 将两个特征在通道维度上拼接
        attn = torch.cat([attn1, attn2], dim=1)


        # 计算平均注意力特征
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 计算最大注意力特征
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 拼接平均和最大注意力特征
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # 通过卷积生成注意力权重，并应用sigmoid激活函数
        sig = self.conv_squeeze(agg).sigmoid()
        # 根据注意力权重调整特征
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 最终卷积恢复到原始通道数
        attn = self.conv(attn)

        # 通过注意力特征加权原输入
        return x * attn

class LSCA(nn.Module):
    def __init__(self, channels ):
        super(LSCA, self).__init__()
        self.LSK1 = LSK1(channels)
        self.CAA = CAA(channels)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        x1 = self.LSK1(x) #C:32
        x2 = self.CAA(x)   #C:32
        a = self.sigmod(x1 + x2)
        out = x + a * x1 + (1 - a) * x2
        return out
