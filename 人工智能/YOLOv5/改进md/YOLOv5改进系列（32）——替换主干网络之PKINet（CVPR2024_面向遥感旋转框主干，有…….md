![](https://i-blog.csdnimg.cn/blog_migrate/8d65a1a503d26620f6fa4568ed0befff.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/5530c8f1cbbf436a257d22aa7035fc4f.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

[YOLOv5改进系列（4）——添加ECA注意力机制][YOLOv5_4_ECA]

[YOLOv5改进系列（5）——替换主干网络之 MobileNetV3][YOLOv5_5_ MobileNetV3]

[YOLOv5改进系列（6）——替换主干网络之 ShuffleNetV2][YOLOv5_6_ ShuffleNetV2]

[YOLOv5改进系列（7）——添加SimAM注意力机制][YOLOv5_7_SimAM]

[YOLOv5改进系列（8）——添加SOCA注意力机制][YOLOv5_8_SOCA]

[YOLOv5改进系列（9）——替换主干网络之EfficientNetv2][YOLOv5_9_EfficientNetv2]

[YOLOv5改进系列（10）——替换主干网络之GhostNet][YOLOv5_10_GhostNet]

[YOLOv5改进系列（11）——添加损失函数之EIoU、AlphaIoU、SIoU、WIoU][YOLOv5_11_EIoU_AlphaIoU_SIoU_WIoU]

[YOLOv5改进系列（12）——更换Neck之BiFPN][YOLOv5_12_Neck_BiFPN]

[YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等][YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]

[YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS][YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）][YOLOv5_20_BiFormer_CVPR2023]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT]

[YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）][YOLOv5_23_MobileViTv2_ Transformer]

[YOLOv5改进系列（24）——替换主干网络之MobileViTv3（移动端轻量化网络的进一步升级）][YOLOv5_24_MobileViTv3]

[YOLOv5改进系列（25）——添加LSKNet注意力机制（大选择性卷积核的领域首次探索）][YOLOv5_25_LSKNet]

[YOLOv5改进系列（26）——添加RFAConv注意力卷积（感受野注意力卷积运算）][YOLOv5_26_RFAConv]

[YOLOv5改进系列（27）——添加SCConv注意力卷积（CVPR 2023|即插即用的高效卷积模块）][YOLOv5_27_SCConv_CVPR 2023]

[YOLOv5改进系列（28）——添加DSConv注意力卷积（ICCV 2023|用于管状结构分割的动态蛇形卷积）][YOLOv5_28_DSConv_ICCV 2023]

[YOLOv5改进系列（29）——添加DilateFormer（MSDA）注意力机制（中科院一区顶刊|即插即用的多尺度全局注意力机制）][YOLOv5_29_DilateFormer_MSDA]

[YOLOv5改进系列（30）——添加iRMB注意力机制（ICCV 2023|即插即用的反向残差注意力机制）][YOLOv5_30_iRMB_ICCV 2023]

[YOLOv5改进系列（31）——添加Dual-ViT注意力机制（TPAMI 2023|京东提出多尺度双视觉Transformer，降低计算开销）][YOLOv5_31_Dual-ViT_TPAMI 2023_Transformer]

![](https://i-blog.csdnimg.cn/blog_migrate/2fd638cb6d5e991c328452d9bd850c95.gif)

目录

[🚀 一、PKINet介绍 ][_PKINet_]

[1.1 PKINet简介][1.1 PKINet]

[1.2 PKINet方法][1.2 PKINet]

[1.2.1 PKI模块][1.2.1 PKI]

[1.2.1 上下文锚点注意力（CAA）][1.2.1 _CAA]

[🚀二、具体添加方法][Link 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在models文件下，创建PKINet.py][models_PKINet.py]

[ 第②步：修改yolo.py文件 ][_yolo.py_]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/3bbaaddf0a85102f3a6bfa5319e7d6ca.gif)

## 🚀 一、PKINet介绍 

> 学习资料：
> 
>  *  论文题目：《Poly Kernel Inception Network for Remote Sensing Detection》
>  *  论文地址：[https://export.arxiv.org/pdf/2403.06258][https_export.arxiv.org_pdf_2403.06258]
>  *  源码地址：[https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet][https_github.com_NUST-Machine-Intelligence-Laboratory_PKINet]

### 1.1 PKINet简介 

背景

遥感图像中目标检测面临的主要挑战包括：

1.  目标尺度变化大
2.  多样的上下文环境

传统方法：

通过扩大网络的空间感受野来解决这些问题，

不足：

大核卷积容易引入背景噪声，而空洞卷积可能生成过于稀疏的特征。

本文提出的方法：

1.  多尺度卷积核：使用无空洞的多尺度卷积核来提取不同尺度的目标特征，这有助于捕获不同尺度下的目标信息，并且能够有效地捕获局部上下文。
2.  Context Anchor Attention（CAA）模块：并行引入CAA模块，用于捕获长距离的上下文信息。这个模块有助于提高目标检测的精度，特别是在处理多样的上下文环境时。

![](https://i-blog.csdnimg.cn/blog_migrate/926d9af6f057a21b7407bd53b90169d1.png)

### 1.2 PKINet方法 

PKINet 设计为特征提取的骨干网络，类似于VGG和ResNet，由四个阶段组成。

每个阶段采用了交叉阶段部分（CSP）结构，其中输入被分成两条路径：

 *  一条路径是简单的前馈网络（FFN）。
 *  另一条路径包括一系列PKI块，每个块由PKI模块和CAA模块组成。

这两条路径的输出被拼接在一起，形成每个阶段的输出。

![](https://i-blog.csdnimg.cn/blog_migrate/2eec1178f3288181a5fca388b8685c1e.png)

#### 1.2.1 PKI模块 

PKI模块是一种Inception风格的模块，它首先通过一个小核卷积来捕获局部信息，然后利用一系列并行的深度卷积来跨多个尺度捕获上下文信息。

第l阶段中第n个PKI块内的PKI模块可以数学地表示如下：

![](https://i-blog.csdnimg.cn/blog_migrate/35f8e6a5a6cdd897a2c162ceb24ffdac.png)

然后，通过大小为1×1的卷积来融合局部和上下文特征，表征不同通道之间的相互关系：

![](https://i-blog.csdnimg.cn/blog_migrate/0f3b739994e1a9d53f71d5eb2bf82fb7.png)

#### 1.2.1 上下文锚点注意力（CAA） 

为了捕获长距离的上下文信息，本文进一步在PKI块中集成了上下文锚点注意力（CAA）模块。CAA旨在把握远距离像素之间的上下文相互依赖性，同时增强中心特征。

以第l阶段中第n个PKI块内的CAA模块为例，采用平均池化后接一个1×1卷积来获取局部区域特征：

![](https://i-blog.csdnimg.cn/blog_migrate/2059747c5e1b783df2daea633e72c309.png)

然后，使用两个深度可分离卷积作为标准大核深度卷积的近似：

![](https://i-blog.csdnimg.cn/blog_migrate/cda5b935abbfe5c95eadd7b68a0bf43f.png)

通过这种方式，CAA模块能够增强模型对远距离依赖性的建模能力，从而提升遥感目标检测的性能。

最后，我们的CAA模块生成一个注意力权重![A_{l-1}, n\in R^{\frac{1}{2}C_{l}\times W_{l}\times H_{l}}](https://latex.csdn.net/eq?A_%7Bl-1%7D%2C%20n%5Cin%20R%5E%7B%5Cfrac%7B1%7D%7B2%7DC_%7Bl%7D%5Ctimes%20W_%7Bl%7D%5Ctimes%20H_%7Bl%7D%7D)，该权重进一步用于增强PKI模块的输出（参见式(4)）：

![](https://i-blog.csdnimg.cn/blog_migrate/0a9ec3a65fbadfa8fe20b4a5ea214572.png)

第l阶段中第n个PKI块的输出是通过以下方式获得的：

![](https://i-blog.csdnimg.cn/blog_migrate/aed49ecdb2e183ad24abf637e3a4ff37.png)

## 🚀二、具体添加方法 

### 2.1 添加顺序 

（1）models/PKINet.py -->  创建新文件，新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，引入PKINet  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_PKINet.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在models文件下，创建PKINet.py 

![](https://i-blog.csdnimg.cn/blog_migrate/94da8d4b1778686155ab498c1cb27d4e.png)

在github上找到对应源码文件，做如下修改：

```java
import math
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init
from mmengine.logging import MMLogger

from models.common import C3,Conv
# from mmrotate.models.builder import ROTATED_BACKBONES
# from utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC

 
def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.
    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa
    Args:
        value (int, float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number.
    """
 
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
 
class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()
 
    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])
 
 
class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()
 
    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])
    
class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
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


class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x


class Stem(BaseModule):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.down_conv = ConvModule(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.conv2(self.conv1(self.down_conv(x)))


class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class PKIBlock(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class PKIStage(BaseModule):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        self.conv1 = ConvModule(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True, norm_cfg=None, act_cfg=None) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity, norm_cfg, act_cfg) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z
```

直接复制上述代码，粘贴到新建的PKINet.py 

> 可能遇到的问题一：ModuleNotFoundError: No module named 'mmcv'
> 
> ![](https://i-blog.csdnimg.cn/blog_migrate/7df235f431d74b3f8db3143e899c4602.png)
> 
> 解决方法：
> 
> （1）第一步：更新pip
> 
> ```java
> python -m ensurepip --upgrade
> ```
> 
> （2）第二步：更新setuptools
> 
> ```java
> pip uninstall -y setuptools 
> pip install setuptools 
> ```
> 
> （3）第三步：根据官方文档选择命令安装
> 
> ```java
> ​​​​​​Installation — mmcv 2.1.0 文档
> ```
> 
> 比如：
> 
> ```java
> pip install -U openmim
> mim install mmcv
> pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
> ```

> 可能遇到的问题二：AssertionError: if use autopad, kernel size must be odd![](https://i-blog.csdnimg.cn/blog_migrate/6b95fb9ee8e6ceff2388a929d40fcffe.png) 之前我们用的方法一直是在common.py尾部加入新的网络结构，然后再yolo.py表中注册。但是这次用同样的方法一示卷积核尺寸必须为奇数（可是我的卷积核尺寸本来就是奇数。。。这里查了好多资料都不知道为啥），所以就用新建文件的方法了。

#### 第②步：修改yolo.py文件 

在yolo.py头部加入这句，引入PKINet

```java
from models.PKINet import PKIBlock
```

#### 第③步：创建自定义的yaml文件 

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone + three Attention modules
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],    # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],    # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, PKIBlock, [256]],    # ---> args改成上一层的通道数
   [-1, 1, Conv, [512, 3, 2]],    # 6-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, PKIBlock, [512]],    # ---> args改成上一层的通道数
   [-1, 1, Conv, [1024, 3, 2]],   # 9-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],      # 11
   [-1, 1, PKIBlock, [1024]],    # ---> args改成上一层的通道数
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 8], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 16

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 5], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 20 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 17], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 23 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 13], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 26 (P5/32-large)

   [[20, 23, 26], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/0577b6a934ca7778981d2bb78d6f2564.png)

这样就OK啦！

## 🌟本人YOLOv5系列导航 

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀[YOLOv5源码][YOLOv5 1]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀[YOLOv5入门实践][YOLOv5 1]系列：

[YOLOv5入门实践（1）——手把手带你环境配置搭建][YOLOv5_1 1]

[YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集][YOLOv5_2_labelimg]

[YOLOv5入门实践（3）——手把手教你划分自己的数据集][YOLOv5_3]

[YOLOv5入门实践（4）——手把手教你训练自己的数据集][YOLOv5_4]

[YOLOv5入门实践（5）——从零开始，手把手教你训练自己的目标检测模型（包含pyqt5界面）][YOLOv5_5_pyqt5]

![](https://i-blog.csdnimg.cn/blog_migrate/f029c0617e18347a611e64e684734e78.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[YOLOv5_5_ MobileNetV3]: https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501
[YOLOv5_6_ ShuffleNetV2]: https://blog.csdn.net/weixin_43334693/article/details/131008642?spm=1001.2014.3001.5501
[YOLOv5_7_SimAM]: https://blog.csdn.net/weixin_43334693/article/details/131031541?spm=1001.2014.3001.5501
[YOLOv5_8_SOCA]: https://blog.csdn.net/weixin_43334693/article/details/131053284?spm=1001.2014.3001.5501
[YOLOv5_9_EfficientNetv2]: https://blog.csdn.net/weixin_43334693/article/details/131207097?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22131207097%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_10_GhostNet]: https://blog.csdn.net/weixin_43334693/article/details/131235113?spm=1001.2014.3001.5501
[YOLOv5_11_EIoU_AlphaIoU_SIoU_WIoU]: https://blog.csdn.net/weixin_43334693/article/details/131350224?spm=1001.2014.3001.5501
[YOLOv5_12_Neck_BiFPN]: https://blog.csdn.net/weixin_43334693/article/details/131461294?spm=1001.2014.3001.5501
[YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]: https://blog.csdn.net/weixin_43334693/article/details/131513850?spm=1001.2014.3001.5502
[YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]: https://blog.csdn.net/weixin_43334693/article/details/131552028?spm=1001.2014.3001.5501
[YOLOv5_15]: https://blog.csdn.net/weixin_43334693/article/details/131613721?spm=1001.2014.3001.5502
[YOLOv5_16_EMA_ICASSP2023]: https://blog.csdn.net/weixin_43334693/article/details/131973273?spm=1001.2014.3001.5501
[YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]: https://blog.csdn.net/weixin_43334693/article/details/131999141?spm=1001.2014.3001.5501
[YOLOv5_18_Neck_AFPN_PAFPN]: https://blog.csdn.net/weixin_43334693/article/details/132070079?spm=1001.2014.3001.5501
[YOLOv5_19_Swin TransformerV1_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132161488?spm=1001.2014.3001.5501
[YOLOv5_20_BiFormer_CVPR2023]: https://blog.csdn.net/weixin_43334693/article/details/132203200?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132203200%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132367429%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502
[YOLOv5_24_MobileViTv3]: https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502
[YOLOv5_25_LSKNet]: https://blog.csdn.net/weixin_43334693/article/details/135510571?spm=1001.2014.3001.5501
[YOLOv5_26_RFAConv]: https://blog.csdn.net/weixin_43334693/article/details/135562865?spm=1001.2014.3001.5501
[YOLOv5_27_SCConv_CVPR 2023]: https://blog.csdn.net/weixin_43334693/article/details/135610505?spm=1001.2014.3001.5502
[YOLOv5_28_DSConv_ICCV 2023]: https://blog.csdn.net/weixin_43334693/article/details/135758781?spm=1001.2014.3001.5502
[YOLOv5_29_DilateFormer_MSDA]: https://blog.csdn.net/weixin_43334693/article/details/135845841?spm=1001.2014.3001.5501
[YOLOv5_30_iRMB_ICCV 2023]: https://blog.csdn.net/weixin_43334693/article/details/139153395?spm=1001.2014.3001.5502
[YOLOv5_31_Dual-ViT_TPAMI 2023_Transformer]: https://blog.csdn.net/weixin_43334693/article/details/139834875?spm=1001.2014.3001.5501
[_PKINet_]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81iRMB%E4%BB%8B%E7%BB%8D%C2%A0
[1.1 PKINet]: #1.1%C2%A0PKINet%E7%AE%80%E4%BB%8B
[1.2 PKINet]: #1.2%C2%A0PKINet%E6%96%B9%E6%B3%95
[1.2.1 PKI]: #1.2.1%C2%A0PKI%E6%A8%A1%E5%9D%97
[1.2.1 _CAA]: #1.2.1%20%E4%B8%8A%E4%B8%8B%E6%96%87%E9%94%9A%E7%82%B9%E6%B3%A8%E6%84%8F%E5%8A%9B%EF%BC%88CAA%EF%BC%89
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[models_PKINet.py]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[_yolo.py_]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9yolo.py%E6%96%87%E4%BB%B6%C2%A0
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0
[Link 2]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_export.arxiv.org_pdf_2403.06258]: https://export.arxiv.org/pdf/2403.06258
[https_github.com_NUST-Machine-Intelligence-Laboratory_PKINet]: https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet
[YOLOv5 1]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501
[YOLOv5_1 1]: https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501
[YOLOv5_2_labelimg]: https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501
[YOLOv5_3]: https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501
[YOLOv5_4]: https://blog.csdn.net/weixin_43334693/article/details/130043351?spm=1001.2014.3001.5501
[YOLOv5_5_pyqt5]: https://blog.csdn.net/weixin_43334693/article/details/130044342?spm=1001.2014.3001.5501