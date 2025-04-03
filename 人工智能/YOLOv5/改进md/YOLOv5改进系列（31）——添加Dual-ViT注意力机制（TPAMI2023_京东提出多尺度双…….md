![](https://i-blog.csdnimg.cn/blog_migrate/097dd38a4ad3c090326136a777179b11.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/30255c03f761aa5ad316172ac8cfd81f.png)

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

[YOLOv5改进系列（28）——添加DSConv注意力卷积（ICCV 2023|用于管状结构分割的动态蛇形卷积）][YOLOv5_28_DSConv_ICCV 2023][YOLOv5改进系列（29）——添加DilateFormer（MSDA）注意力机制（中科院一区顶刊|即插即用的多尺度全局注意力机制）][YOLOv5_29_DilateFormer_MSDA]

[YOLOv5改进系列（30）——添加iRMB注意力机制（ICCV 2023|即插即用的反向残差注意力机制）][YOLOv5_30_iRMB_ICCV 2023]

![](https://i-blog.csdnimg.cn/blog_migrate/2c781cfaf07b33d2a3295d0f6392a564.gif)

目录

[🚀 一、Dual-ViT介绍 ][_Dual-ViT_]

[1.1 Dual-ViT简介 ][1.1 Dual-ViT_]

[ 1.2 Dual-ViT方法 ][1.2 Dual-ViT_]

[（1）Dual Block—双重块][1_Dual Block]

[（2）Merge Block—合并块][2_Merge Block]

[（3）Dual Vision Transformer—双视觉Transformer ][3_Dual Vision Transformer_Transformer]

[🚀二、具体添加方法 ][Link 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加DualViT模块 ][common.py_DualViT_]

[第②步：修改yolo.py文件 ][yolo.py_]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/294f3b5ab9f36e04dab421e8c3b36a19.gif)

## 🚀 一、Dual-ViT介绍 

> 学习资料：
> 
>  *  论文题目：《Dual Vision Transformer》
>  *  论文地址：[https://arxiv.org/abs/2207.04976][https_arxiv.org_abs_2207.04976]
>  *  源码地址：[GitHub - YehLi/ImageNetModel: Official ImageNet Model repository][GitHub - YehLi_ImageNetModel_ Official ImageNet Model repository]

### 1.1 Dual-ViT简介 

先前工作及存在的问题

 *  先前工作提出多种降低自注意力计算成本的策略，如分解为区域和局部特征提取，显著减少计算复杂度。
 *  但这些方法通常因为下采样导致区域信息丢失，不可避免地损失了一些关键信息。

本文介绍的新型Transformer架构：双视觉Transformer（Dual ViT）

 *  目的是解决现有方法中的计算成本和信息损失问题。
 *  结合关键的语义路径，有效地将token向量压缩为全局语义，从而降低复杂度。
 *  引入像素路径，利用像素级细节作为有用的先验信息，以补充全局语义的压缩。
 *  通过联合训练整合语义路径和像素路径，增强自注意力信息的传播效果。

实验结果

 *  实验证明，双ViT相比当前领先的Transformer架构（SOTA）具有更高的精度，同时训练复杂度更低。

### 1.2 Dual-ViT方法 

Dual-ViT采用了新的双块，由两个路径组成：

1.  语义路径，在全局级抽象高级语义token。
2.  像素路径，通过在像素级重新定义输入特征来捕获细粒度信息。

#### （1）Dual Block—双重块 

![](https://i-blog.csdnimg.cn/blog_migrate/be463a61c62627c3ef52d45eb76f30ce.png)

两个路径的作用：

 *  语义路径：将输入特征映射总结为语义token。
 *  像素路径：在此之后，以键/值的形式优先考虑这些语义token，并通过交叉注意力对定义的输入特征图进行多头注意力。

复杂性分析：

由于语义路径包含的token比像素路径中的token少得多，因此计算成本降低到O(nmd + m^2~d)

其中m是语义token的数量。

目的：

考虑到梯度通过两种路径反向传播，双重块算法能够同时补偿全局特征压缩的信息损失，并通过语义-像素交互降低全局先验局部特征提取的难度。

这种新设计很好地引入了一种额外的途径，可以通过全局语义信息来促进自我注意学习。

#### （2）Merge Block—合并块 

![](https://i-blog.csdnimg.cn/blog_migrate/72bddb64576275b53c490add3ce76724.png)

解决的问题：

双块利用了语义路径和像素路径之间的相互作用。然而，高分辨率输入复杂性下，像素路径中局部token的内部相互作用未充分利用。

本文提出解决方法：

 *  作者提出了一种称为合并块的自注意力块设计。
 *  设计目的是在最后两个阶段（使用低分辨率输入）对合并了的语义和局部token执行自注意力操作，以促进局部token之间的内部交互。

#### （3）Dual Vision Transformer—双视觉Transformer  

![](https://i-blog.csdnimg.cn/blog_migrate/a800e92f419ee006512ea4356f3f973c.png)完整的Dual-ViT包含四个阶段：

前两个阶段由Dual块堆栈组成，后两个阶段由合并块组成。

## 🚀二、具体添加方法 

### 2.1 添加顺序 

（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将iRMB类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_DualViT.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加DualViT模块  

将下面的DualViT代码复制粘贴到common.py文件的末尾。

```java
#DualViT
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
 
# ========== 1.DWConv类：深度可分离卷积 ==========
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
 
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
 
# ========== 2.PVT2FFN类：组合了线性层、深度可分离卷积层和激活函数，适用于处理输入特征并输出经过一系列线性和非线性变换后的结果。 ========== 
class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
 
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x
 
# ========== 3.MergeFFN类：特征融合和转换 ========== 
class MergeFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
 
        self.fc_proxy = nn.Sequential(
            nn.Linear(in_features, 2 * in_features),
            nn.GELU(),
            nn.Linear(2 * in_features, in_features),
        )
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
 
    def forward(self, x, H, W):
        x, semantics = torch.split(x, [H * W, x.shape[1] - H * W], dim=1)
        semantics = self.fc_proxy(semantics)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)
        return x
 
# ========== 4.Attention类：注意力机制 ========== 
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
 
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
 
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
 
    def forward(self, x):
       # x =x.permute(3, 0, 1, 2)
        B, H, W, C = x.shape
 
        N = H * W
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W , C)
        x = self.proj(x)
        return x
 
 
 
# ========== 5.MergeBlockattention类：本文核心方法 ========== 
class MergeBlockattention(nn.Module):
    def __init__(self,input, dim, num_heads=2, mlp_ratio=8, drop_path=0., norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
       '''
        self.norm1: 第一个归一化层 (nn.LayerNorm)，对输入数据进行归一化处理。
        self.norm2: 第二个归一化层，同样用于归一化处理。
        self.attn: 注意力机制模块 (Attention)，用于计算输入的注意力表示。
        self.mlp: 根据 is_last 参数选择不同的多层感知机模块。如果 is_last 为 True，则使用 
        PVT2FFN；否则使用之前定义的 MergeFFN。
        self.is_last: 标志位，指示是否是最后一个模块。
        self.drop_path: 如果指定了 drop_path 大于 0，则应用一定概率的 DropPath，否则使用恒等 
                        映射 (nn.Identity()).
         self.gamma1, self.gamma2: 可学习的参数 gamma，用于调节归一化后的特征。
       '''
        if is_last:
            self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        else:
            self.mlp = MergeFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.is_last = is_last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
 
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        #x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x =self.attn(self.norm1(x))
 
        x = x.permute(0, 3, 2, 1)
 
        return x
```

#### 第②步：修改yolo.py文件 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/8e93b404945ea6a835e56b114fa271fd.png)

加入 MergeBlockattention 这个模块

```java
if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, MergeBlockattention]:
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
   [-1, 1, MergeBlockattention, [256]],    # ---> args改成上一层的通道数
   [-1, 1, Conv, [512, 3, 2]],    # 6-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, MergeBlockattention, [512]],    # ---> args改成上一层的通道数
   [-1, 1, Conv, [1024, 3, 2]],   # 9-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],      # 11
   [-1, 1, MergeBlockattention, [1024]],    # ---> args改成上一层的通道数
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

![](https://i-blog.csdnimg.cn/blog_migrate/3e41e0549c9bf50b6b15fe19a1f6e105.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/1326a0a491970540a402d5482fb12db3.gif)


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
[YOLOv5_30_iRMB_ICCV 2023]: https://blog.csdn.net/weixin_43334693/article/details/139153395?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22139153395%22%2C%22source%22%3A%22weixin_43334693%22%7D
[_Dual-ViT_]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81Dual-ViT%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0
[1.1 Dual-ViT_]: #1.1%C2%A0Dual-ViT%E7%AE%80%E4%BB%8B%C2%A0
[1.2 Dual-ViT_]: #%C2%A01.2%C2%A0Dual-ViT%E6%96%B9%E6%B3%95%C2%A0
[1_Dual Block]: #%EF%BC%881%EF%BC%89Dual%20Block%E2%80%94%E5%8F%8C%E9%87%8D%E5%9D%97
[2_Merge Block]: #%EF%BC%882%EF%BC%89Merge%20Block%E2%80%94%E5%90%88%E5%B9%B6%E5%9D%97
[3_Dual Vision Transformer_Transformer]: #%EF%BC%883%EF%BC%89Dual%20Vision%20Transformer%E2%80%94%E5%8F%8C%E8%A7%86%E8%A7%89Transformer%C2%A0
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95%C2%A0
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_DualViT_]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[yolo.py_]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9yolo.py%E6%96%87%E4%BB%B6%C2%A0
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0%C2%A0%C2%A0%C2%A0
[Link 2]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2207.04976]: https://arxiv.org/abs/2207.04976
[GitHub - YehLi_ImageNetModel_ Official ImageNet Model repository]: https://github.com/YehLi/ImageNetModel
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