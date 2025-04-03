![](https://i-blog.csdnimg.cn/blog_migrate/1db73bbbb6836f0418b4d65c6d28c7a0.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/46f1e42835c99645ee7d6db409e7e44d.png)

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

[YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]

[YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

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

![](https://i-blog.csdnimg.cn/blog_migrate/b246425d686358b7074a3c33aaabd7d7.gif)

目录

[🚀 一、iRMB介绍 ][_iRMB_]

[1.1 iRMB简介][1.1 iRMB]

[1.2 iRMB结构][1.2 iRMB]

[元移动模块][Link 1]

[反向残差模块][Link 2]

[🚀二、具体添加方法][Link 3]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加iRMB模块 ][common.py_iRMB_]

[第②步：修改yolo.py文件 ][yolo.py_]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 4]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/6d7832f0e74b9e24a07416e900369a17.gif)

## 🚀 一、iRMB介绍 

>  学习资料：
> 
>  *  论文题目：《Rethinking Mobile Block for Efficient Attention-based Models》
>  *  论文地址：[https://arxiv.org/pdf/2301.01146.pdf][https_arxiv.org_pdf_2301.01146.pdf]
>  *  源码地址：[GitHub - zhangzjn/EMO: \[ICCV 2023\] Official PyTorch implementation of "Rethinking Mobile Block for Efficient Attention-based Models"][GitHub - zhangzjn_EMO_ _ICCV 2023_ Official PyTorch implementation of _Rethinking Mobile Block for Efficient Attention-based Models]

2023 腾讯优图/浙大/北大提出：重新思考高效神经模型的移动模块

### 1.1 iRMB简介 

背景 

 *  基于Depth-Wise Convolution的IRB成为轻量化卷积模型的基础构成模块，但受限于静态CNN归纳偏置影响，这些模型的效果（尤其是下游检测、分割等任务）仍有待进一步提高。
 *  得益于Transformer的动态建模能力，基于ViT（Vison Transformer）的许多工作（如DeiT、PVT、SwinTransformer等）较CNN模型取得了明显的效果提升。然而，受限于MHSA（Multi-Head Self-Attention，多头自注意力）的二次方计算量需求，Transformer很少应用于移动端场景。

之前的尝试

因此一些研究人员尝试设计融合上述两者的混合模型，但一般会引入复杂的结构或多个混合模块，

本文的方法

这篇文章旨在设计轻量级高效的模型结构，平衡了参数量、计算量和精度.

本文首先抽象了MMB（Meta Mobile Block）用来对IRB和Transformer中的MHSA(Multi-Head Self-Attention，多头自注意力）/FFN(Feed-Forward Network，前向模型）进行归纳，其次实例化了高效的iRMB（Inverted Residual Mobile Block），最后仅使用该模块构建了高效的EMO（Efficient MOdel）轻量化主干模型。

实现效果

 *  1M/2M/5M 尺度下的 EMO 模型在 ImageNet-1K 的分类数据集上分别获得了 71.5/75.1/78.4 Top-1精度，超过了当前 SoTA 基于 CNN/Transformer 的轻量化模型；
 *  基于 SSDLite 框架仅使用 0.6G/0.9G/1.8G FLOPs 获得了 22.0/25.2/27.9 mAP；
 *  基于 DeepLabv3框架仅使用 2.4G/3.5G/5.8G FLOPs 获得了 33.5/35.3/37.8 mIoU，取了较好的 mobile 条件下的效果。

![](https://i-blog.csdnimg.cn/blog_migrate/d042dd9fb1040871f030fb9cf83baa47.png)

### 1.2 iRMB结构 

iRMB结合了CNN的轻量级特性和Transformer模型的动态处理能力，这种结构特别适用于移动设备上需要进行密集预测任务的场景，因为它旨在在计算资源有限的情况下提供高效性能。

![](https://i-blog.csdnimg.cn/blog_migrate/c9cb75f65a859dcf8be12b155e3ba0db.png)

图2：左侧：抽象的统一元移动块，来源于多头自注意力/前馈网络和倒置残差块。归纳块可以通过不同的扩张比例λ和高效运算符F 推导成具体模块。右侧：类似ResNet的EMO，只由推导出的iRMB组成。

在图2中归纳出一个通用的Meta Mobile块（MMB），其中采用参数化参数扩张比例λ和高效运算符F 来实例化不同模块。

#### 元移动模块 

与MetaFormer的关系

![](https://i-blog.csdnimg.cn/blog_migrate/28a8e3ab36cc60a740c11e1d3b777006.png)

基于Meta Mobile Block，设计了一个反向残差移动块 (iRMB)，它吸收了 CNN 架构的效率来建模局部特征和 Transformer 架构动态建模的能力来学习长距离交互。

如上图左侧所示，通过对 MobileNetv2 中的 IRB 以及 Transformer 中的核心 MHSA 和 FFN 模块进行抽象，本文提出了统一的 MMB 对上述几个结构进行归纳表示，即采用扩张率 𝜆 和高效算子 𝐹 来实例化不同的模块。

#### 反向残差模块 

反向残差模块（Inverted Residual Block，IRB）是一种用于深度神经网络中的模块结构。在iRMB添加倒置残差块（IRB）得模型能够更有效地处理长距离信息。

> 什么是倒残差模块？
> 
> 通常情况下，残差连接会将输入添加到层的输出中，而在倒置残差块中，先将输入通过一个轻量级的层（比如1x1卷积）进行降维，然后再将降维后的特征进行处理，并最终将处理后的结果与原始输入相加，形成残差连接。这样的设计可以减少计算量和参数数量，同时仍然可以保持模型的性能和效率。

在iRMB中基于Meta Mobile Block设计一个反向残差移动块 (iRMB)，它吸收了 CNN 架构的效率来建模局部特征和 Transformer 架构动态建模的能力来学习长距离交互。反向残差块通常被用于轻量级的模型设计中，以提高模型的效率和推理速度。

![](https://i-blog.csdnimg.cn/blog_migrate/c029cabe0e333c4d107a4f6491c0262b.png)

不同模型的效果主要来源于高效算子 𝐹 的具体形式，考虑到轻量化和易用性，本文将 MMB 中的 𝐹 建模为Expanded Window MHSA（EW-MHSA）和Depth-Wise Convolution（DW-Conv）的级联，兼顾动态全局建模和静态局部信息融合的优势，同时能够有效地增加模型的感受野，提升对于下游任务的能力。

![](https://i-blog.csdnimg.cn/blog_migrate/f08b7a4fe9dc00f20f67e060ef8207dd.png)

进一步，作者将 𝐹 设置为 4 并替换 DeiT 和 PVT 中标准 Transformer 结构以评估 iRMB 性能，如表3所述，可以发现iRMB可以在相同的训练设置下以更少的参数和计算获得更高的性能。

![](https://i-blog.csdnimg.cn/blog_migrate/e7a17e436021c7c8c464263d81f68f9c.png)

## 🚀二、具体添加方法 

### 2.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将iRMB类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml --> 新建一个文件夹，如yolov5s\_iRMB.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加iRMB模块  

将下面的iRMB代码复制粘贴到common.py文件的末尾。

```java
# iRMB
import math
from functools import partial
from einops import rearrange
from timm.models.layers.activations import *
from timm.models.layers import DropPath
from timm.models.efficientnet_builder import _parse_ksize
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE
 
# ========== 1.LayerNorm2d类：实现对输入张量进行二维的 Layer Normalization 操作 ==========
class LayerNorm2d(nn.Module):
 
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
 
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x
 
 
def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]
 
def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]

# ========== 2.ConvNormAct类：实现卷积、规范化和激活操作的集合 ==========
class ConvNormAct(nn.Module):
 
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
 
    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
 
# ========== 3.iRMB类：反向残差注意力机制 ========== 
class iRMB(nn.Module):
 
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False,inplace=True):
'''
   dim_in: 输入特征的维度。
   dim_out: 输出特征的维度。
   norm_in: 是否对输入进行标准化。
   has_skip: 是否使用跳跃连接。
   exp_ratio: 扩展比例。
   norm_layer: 标准化层的类型。
   act_layer: 激活函数的类型。
   v_proj: 是否对V进行投影。
   dw_ks: 深度可分离卷积的卷积核大小。
   stride: 卷积的步幅。
   dilation: 卷积的膨胀率。
   se_ratio: SE 模块的比例。
   dim_head: 注意力头的维度。
   window_size: 窗口大小。
   attn_s: 是否使用注意力机制。
   qkv_bias: 是否在注意力机制中使用偏置。
   attn_drop: 注意力机制中的dropout比例。
   drop: 全连接层的dropout比例。
   drop_path: DropPath 的比例。
   v_group: 是否对 V 进行分组卷积。
   attn_pre: 是否将注意力机制应用到输入之前。
   inplace: 是否原地执行操作。
'''

        super().__init__()  # 调用父类的构造函数
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()  # 条件判断，返回一个标准化层（例如 BatchNorm、LayerNorm 等）或使用空操作
        dim_mid = int(dim_in * exp_ratio)  # 计算中间维度大小
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip  # 条件判断，是否使用跳跃连接
        self.attn_s = attn_s  # 是否使用空间注意力机制的标志

        # 如果使用注意力机制
        if self.attn_s:  
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'  # 确保输入维度 dim_in 可以被 dim_head 整除
            self.dim_head = dim_head  # 设置每个头的维度为 dim_head
            self.window_size = window_size  # 设置窗口大小
            self.num_head = dim_in // dim_head  # 计算头数 self.num_head
            self.scale = self.dim_head ** -0.5  # 计算缩放因子 self.scale，用于调节注意力分数
            self.attn_pre = attn_pre  # 设定是否在注意力机制之前重新排列数据 self.attn_pre

            # 创建 QK 卷积层、V 卷积层、注意力机制的 dropout 等
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')  
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)  
            self.attn_drop = nn.Dropout(attn_drop)  

        # 如果不使用注意力机制
        else:  
           # 如果需要进行 V 投影，则创建 V 卷积层；否则使用 nn.Identity() 空操作
            if v_proj:  # 如果使用V投影
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=inplace)  # 创建V卷积层
            else:
                self.v = nn.Identity()  # 使用空操作
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)  # 创建局部卷积层
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()  # 创建空间激励模块或使用空操作
 
        self.proj_drop = nn.Dropout(drop) 
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)  
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()  

def forward(self, x):
    shortcut = x  # 保存输入的快捷连接
    x = self.norm(x)  # 应用标准化层

    # 提取输入 x 的形状信息
    B, C, H, W = x.shape

    if self.attn_s:  # 如果使用了注意力机制
        # padding
        if self.window_size <= 0:
            window_size_W, window_size_H = W, H
        else:
            window_size_W, window_size_H = self.window_size, self.window_size
        # 计算填充的大小
        pad_l, pad_t = 0, 0
        pad_r = (window_size_W - W % window_size_W) % window_size_W
        pad_b = (window_size_H - H % window_size_H) % window_size_H
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))  # 对输入进行填充
        n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
        x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()  # 重新排列输入数据

        # attention
        b, c, h, w = x.shape
        qk = self.qk(x)  # 计算查询和键的表示
        qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                       dim_head=self.dim_head).contiguous()  # 重排查询和键的表示
        q, k = qk[0], qk[1]
        attn_spa = (q @ k.transpose(-2, -1)) * self.scale  # 计算空间注意力矩阵
        attn_spa = attn_spa.softmax(dim=-1)  # 对注意力矩阵进行 softmax
        attn_spa = self.attn_drop(attn_spa)  # 应用注意力 dropout
        if self.attn_pre:
            x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()  # 重排输入特征
            x_spa = attn_spa @ x  # 应用注意力矩阵到输入特征
            x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                              w=w).contiguous()  # 重排输出特征
            x_spa = self.v(x_spa)  # 对输出特征应用值的表示
        else:
            v = self.v(x)  # 计算值的表示
            v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()  # 重排值的表示
            x_spa = attn_spa @ v  # 应用注意力矩阵到值的表示
            x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                              w=w).contiguous()  # 重排输出特征

        # unpadding
        x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()  # 重新排列输出特征
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()  # 移除填充部分

    else:  # 如果不使用注意力机制
        x = self.v(x)  # 计算值的表示

    # 应用空间激励模块和局部卷积层
    x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

    # 应用输出投影的 dropout
    x = self.proj_drop(x)  # 应用 dropout
    x = self.proj(x)  # 应用输出投影

    # 添加快捷连接并应用路径丢弃
    x = (shortcut + self.drop_path(x)) if self.has_skip else x  # 添加快捷连接并应用路径丢弃
    return x  # 返回处理后的结果
```

#### 第②步：修改yolo.py文件 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/015ea2925046c213eeb09ba320482cfe.png)

加入 iRMB这两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/66e61bb39c807234aa0659d86194ab01.png)

#### 第③步：创建自定义的yaml文件 

```java
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
   [-1, 1, iRMB, [256]],    # 修改1
   [-1, 1, Conv, [512, 3, 2]],    # 6-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, iRMB, [512]],    # 修改2
   [-1, 1, Conv, [1024, 3, 2]],   # 9-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],      # 11
   [-1, 1, iRMB, [1024]],    # 修改3
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

![](https://i-blog.csdnimg.cn/blog_migrate/2f17090388bc62009f62d3ceba555611.png)

### 🌟本人YOLOv5系列导航 

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

![](https://i-blog.csdnimg.cn/blog_migrate/9694a1cbbcdde16ea9b9f9a9627f299f.png)


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
[_iRMB_]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81iRMB%E4%BB%8B%E7%BB%8D%C2%A0
[1.1 iRMB]: #1.1%C2%A0iRMB%E7%AE%80%E4%BB%8B
[1.2 iRMB]: #1.2%C2%A0iRMB%E7%BB%93%E6%9E%84
[Link 1]: #h_652608641_3
[Link 2]: #h_652608641_4
[Link 3]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_iRMB_]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[yolo.py_]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9yolo.py%E6%96%87%E4%BB%B6%C2%A0
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0%C2%A0%C2%A0
[Link 4]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_pdf_2301.01146.pdf]: https://arxiv.org/pdf/2301.01146.pdf
[GitHub - zhangzjn_EMO_ _ICCV 2023_ Official PyTorch implementation of _Rethinking Mobile Block for Efficient Attention-based Models]: https://github.com/zhangzjn/EMO
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