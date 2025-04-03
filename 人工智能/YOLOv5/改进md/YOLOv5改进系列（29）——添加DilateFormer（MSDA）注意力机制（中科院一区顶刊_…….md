![](https://i-blog.csdnimg.cn/blog_migrate/dd9c8e56b362b4826c1bafd7f843c29f.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/a0fd89d5bf65f91d235dbef3111468c6.png)

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

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT]

[YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）][YOLOv5_23_MobileViTv2_ Transformer]

[YOLOv5改进系列（24）——替换主干网络之MobileViTv3（移动端轻量化网络的进一步升级）][YOLOv5_24_MobileViTv3]

[YOLOv5改进系列（25）——添加LSKNet注意力机制（大选择性卷积核的领域首次探索）][YOLOv5_25_LSKNet]

[YOLOv5改进系列（26）——添加RFAConv注意力卷积（感受野注意力卷积运算）][YOLOv5_26_RFAConv]

[YOLOv5改进系列（27）——添加SCConv注意力卷积（CVPR 2023|即插即用的高效卷积模块）][YOLOv5_27_SCConv_CVPR 2023]

[YOLOv5改进系列（28）——添加DSConv注意力卷积（ICCV 2023|用于管状结构分割的动态蛇形卷积）][YOLOv5_28_DSConv_ICCV 2023]

![](https://i-blog.csdnimg.cn/blog_migrate/4b4c71ac227176648955d7cff525800e.gif)

目录

[🚀 一、DilateFormer介绍][_DilateFormer]

[1.1 DilateFormer简介 ][1.1 DilateFormer_]

[1.2 DilateFormer网络结构][1.2 DilateFormer]

[① DilateFormer][DilateFormer]

[② MSDA][MSDA]

[🚀二、具体添加方法][Link 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加DilateFormer模块 ][common.py_DilateFormer_]

[第②步：修改yolo.py文件 ][yolo.py_]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/102a36e3355d23b67c415a295a5b2ed3.gif)

## 🚀 一、DilateFormer介绍 

> 学习资料：
> 
>  *  论文题目：《DilateFormer: Multi-Scale Dilated [Transformer][] for Visual Recognition》
>  *  论文地址：[https://arxiv.org/abs/2302.01791][https_arxiv.org_abs_2302.01791]
>  *  源码地址：[https://github.com/abhinavsagar/mssa][https_github.com_abhinavsagar_mssa]

### 1.1 DilateFormer简介 

背景

ViT 模型在计算复杂性和感受野大小之间的权衡上存在矛盾：

 *  具体来说，ViT 模型使用全局注意力机制，能够在任意图像块之间建立长远距离上下文依赖关系，但是全局感受野带来的是平方级别的计算代价。
 *  同时，在浅层特征上，直接进行全局依赖性建模可能存在冗余，因此是没必要的。

本文方法 

（1）多尺度空洞注意力（MSDA） 

 *  MSDA 能够模拟小范围内的局部和稀疏的图像块交互
 *  在浅层次上，注意力矩阵具有局部性和稀疏性两个关键属性
 *  这表明在浅层次的语义建模中，远离查询块的块大部分无关
 *  因此，全局注意力模块中存在大量的冗余。

（2）DilateFormer 

 *  在底层阶段堆叠 MSDA 块
 *  在高层阶段使用全局多头自注意力块
 *  这样的设计使得模型能够在处理低级信息时，充分利用局部性和稀疏性，而在处理高级信息时，又能模拟远距离依赖

实现效果

 *  论文通过在不同的视觉任务上进行实验，发现 DilateFormer 模型取得了很好的效果。
 *  在 ImageNet-1K 分类任务上，与现有最优秀的模型相比，DilateFormer 模型的性能相当，但所需的FLOPs（浮点运算次数）减少了70%。
 *  在其它的视觉任务，如 COCO 对象检测/实例分割任务和 ADE20K 语义分割任务，DilateFormer 模型同样取得了优秀的表现。

### 1.2 DilateFormer网络结构 

#### ① DilateFormer 

![](https://i-blog.csdnimg.cn/blog_migrate/18b7346d7cdd27499733a23d5fb44883.png)

如上图所示，DilateFormer的整体架构主要由四个阶段构成：

 *  在第一阶段和第二阶段，使用 MSDA
 *  在第三阶段和第四阶段，使用普通的多头自注意力（MHSA）

对于图像输入：DilateFormer 首先使用重叠的分词器进行 patch 嵌入，然后通过交替控制卷积核的步长大小（1或2）来调整输出特征图的分辨率。

对于前一阶段的 patches，采用了一个重叠的下采样器，具有重叠的内核大小为 3，步长为 2。

整个模型的每一部分都使用了条件位置嵌入（CPE）来使位置编码适应不同分辨率的输入。

代码：

```java
class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim # 表示输入特征的维度
        self.num_heads = num_heads # 表示多头注意力机制中的注意力头数
        self.mlp_ratio = mlp_ratio # MLP（多层感知机）隐藏层维度与输入维度之比
        self.kernel_size = kernel_size 
        self.dilation = dilation # 卷积核的膨胀率，指定每一层的膨胀率
        self.cpe_per_block = cpe_per_block  # 是否在每个块中使用位置嵌入
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        '''
         : qkv_bias：是否允许注意力机制中的查询、键、值的偏置。
         : qk_scale：注意力机制中查询和键的缩放因子。
         : drop：MLP中的dropout概率。
         : attn_drop：注意力机制中的dropout概率。
        '''
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity() 

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        #B, C, H, W
        return x
```

过程如下： 

1.  首先，如果启用了位置嵌入（`cpe_per_block`为true），则将输入张量与通过`nn.Conv2d`实现的位置嵌入相加。
2.  接着，将输入张量维度进行调整，将通道维度放到最后。
3.  然后，通过多头膨胀局部注意力机制处理输入，再通过MLP处理上一步的输出。
4.  最后，再次调整张量维度，并返回处理后的结果。

#### ② MSDA 

![](https://i-blog.csdnimg.cn/blog_migrate/479722a1272e23618dbec6da26db05bb.png)

MSAD 模块是DilateFormer 的核心部分。

如上图所示，MSDA 模块同样采用多头的设计：将特征图的通道分为 n 个不同的头部，并在不同的头部使用不同的空洞率执行滑动窗口膨胀注意力（SWDA）。

目的：这样可以在被关注的感受野内的各个尺度上聚合语义信息，并有效地减少自注意力机制的冗余，无需复杂的操作和额外的计算成本。

具体的操作如下：

1.  对于每个头部，都会有一个独立的膨胀率![r_{i}](https://latex.csdn.net/eq?r_%7Bi%7D)
2.  从特征图中获取切片![Q_{i}](https://latex.csdn.net/eq?Q_%7Bi%7D)，![K_{i}](https://latex.csdn.net/eq?K_%7Bi%7D)，![V_{i}](https://latex.csdn.net/eq?V_%7Bi%7D)，执行 SWDA，得到输出![h_{i}](https://latex.csdn.net/eq?h_%7Bi%7D)
3.  将所有头部的输出连接在一起，然后通过一个线性层进行特征聚合

代码：

```java
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim # 注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5 # 查询和键的缩放因子，默认为 head_dim ** -0.5。
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop) # 注意力机制中的dropout概率

    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)# B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C//self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])# B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

 class DilateAttention

1.  首先，对输入的`q`、`k`、`v`进行一些预处理。
2.  然后，将`q`和`k`进行矩阵乘法，并使用缩放因子进行缩放，得到的矩阵表示注意力分布。
3.  接着，对注意力分布进行 softmax、dropout 操作，将注意力分布与`v`进行矩阵乘法。
4.  最后，通过一系列的维度调整得到输出张量 `x`。

class MultiDilatelocalAttention 

1.  首先，对输入张量进行维度调整，将通道维度放到第二个位置。
2.  然后，使用一个卷积层 `self.qkv` 对输入张量进行处理，得到包含查询、键、值的张量，再进行维度调整。
3.  接着，使用一个 `nn.ModuleList` 包含多个`DilateAttention`模块，每个模块对输入进行多头膨胀局部注意力机制的处理。
4.  最后，将处理后的张量经过全连接层和dropout层，得到最终的输出。

## 🚀二、具体添加方法 

### 2.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将DilateFormer类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_DilateFormer.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加DilateFormer模块  

将下面的DilateFormer代码复制粘贴到common.py文件的末尾。

```java
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
 
 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
 
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size=kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)
 
    def forward(self,q,k,v):
        #B, C//3, H, W
        B,d,H,W = q.shape
 
        q = q.reshape([B, d//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape([B, d//self.head_dim, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x
 
 
class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"
 
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
 
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
    def forward(self, x):
 
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        B, C, H, W = x.shape
 
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C //self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C//self.num_dilation, H, W).permute(1, 0, 3, 4, 2 )
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])# B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
 
class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
 
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
 
 
 
    def forward(self, x):
 
        x = x.permute(0, 3, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.permute(0, 3, 2, 1)
        #B, C, H, W
        return x
 
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
 
class C3_DilateBlock(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(DilateBlock(c_) for _ in range(n)))
 
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```

#### 第②步：修改yolo.py文件 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/015ea2925046c213eeb09ba320482cfe.png)

加入DilateBlock、C3\_DilateBlock这两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/1eed8944e62bf370e7d2d7a5a37ec3c4.png)

#### 第③步：创建自定义的yaml文件 

第1种，在SPPF前单独加一层

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

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 3, DilateBlock, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9

  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

第2种，替换C3模块

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

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9

  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3_DilateBlock, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3_DilateBlock, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3_DilateBlock, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3_DilateBlock, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

运行yolo.py

第1种

![](https://i-blog.csdnimg.cn/blog_migrate/59b0d2b61f1564b0ec74987ed9042f66.png)第2种

![](https://i-blog.csdnimg.cn/blog_migrate/a19c96ffd03a7cee00642e2d5a3ed206.png) 这样就OK啦！

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

![](https://i-blog.csdnimg.cn/blog_migrate/c38d39259a7080120a8da1fdc2250ac5.png)


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
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132367429%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502
[YOLOv5_24_MobileViTv3]: https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502
[YOLOv5_25_LSKNet]: https://blog.csdn.net/weixin_43334693/article/details/135510571?spm=1001.2014.3001.5501
[YOLOv5_26_RFAConv]: https://blog.csdn.net/weixin_43334693/article/details/135562865?spm=1001.2014.3001.5501
[YOLOv5_27_SCConv_CVPR 2023]: https://blog.csdn.net/weixin_43334693/article/details/135610505?spm=1001.2014.3001.5502
[YOLOv5_28_DSConv_ICCV 2023]: https://blog.csdn.net/weixin_43334693/article/details/135758781?spm=1001.2014.3001.5502
[_DilateFormer]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81DSConv%E4%BB%8B%E7%BB%8D%C2%A0
[1.1 DilateFormer_]: #1.1%C2%A0DilateFormer%E7%AE%80%E4%BB%8B%C2%A0
[1.2 DilateFormer]: #1.2%C2%A0%C2%A0DilateFormer%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[DilateFormer]: #%E2%91%A0%C2%A0DilateFormer
[MSDA]: #%C2%A0%E2%91%A1%20MSDA
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_DilateFormer_]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[yolo.py_]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9yolo.py%E6%96%87%E4%BB%B6%C2%A0
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0%C2%A0%C2%A0
[Link 2]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[Transformer]: https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020
[https_arxiv.org_abs_2302.01791]: https://arxiv.org/abs/2302.01791
[https_github.com_abhinavsagar_mssa]: https://github.com/abhinavsagar/mssa
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