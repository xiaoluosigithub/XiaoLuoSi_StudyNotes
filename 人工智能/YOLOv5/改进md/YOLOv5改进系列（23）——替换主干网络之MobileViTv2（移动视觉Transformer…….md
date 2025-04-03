![](https://i-blog.csdnimg.cn/blog_migrate/9abb5b2338de720ad3812461bcdc7560.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/813905c6e1d7dfc09433e23374a228d0.png)

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

[YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等][YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC][YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS][YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）][YOLOv5_20_BiFormer_CVPR2023]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT]

![](https://i-blog.csdnimg.cn/blog_migrate/b432bc3c0686ff75fa31365d201e1c46.gif)

目录

[🚀一、MobileViT v2介绍 ][MobileViT v2_]

[1.1 简介 ][1.1 _]

[1.2 网络结构][1.2]

[MHA][]

[Separable self-attention][]

[1.3 实验][1.3]

[🚀二、具体添加方法 ][Link 1]

[第①步：在common.py中添加MobileViTv2模块][common.py_MobileViTv2]

[第②步：修改yolo.py文件][yolo.py]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步 验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/4b4a6c296e738e68b3d860911546a180.gif)

## 🚀一、MobileViT v2介绍 

>  *  论文题目：《Separable Self-attention for Mobile Vision Transformers 》
>  *  论文地址：[https://arxiv.org/abs/2110.02178][https_arxiv.org_abs_2110.02178]
>  *  源码地址：[GitHub - apple/ml-cvnets: CVNets: A library for training computer vision networks][GitHub - apple_ml-cvnets_ CVNets_ A library for training computer vision networks]

![](https://i-blog.csdnimg.cn/blog_migrate/00a2a25764bd20b66c601c3065de14e4.png)

### 1.1 简介 

上一篇介绍的MobileViT可以在多个移动视觉任务中实现最先进的性能，包括分类和检测。虽然这些模型的参数较少，但与基于卷积神经网络的模型相比，它们具有较高的延迟。

其主要效率瓶颈：

1.  transformer中的多头自我注意(MHA)，相对于token的数量k，它需要O(k^2)的时间复杂度。
2.  MHA需要昂贵的操作(例如，批量矩阵乘法)来计算自我注意，影响资源受限设备的延迟。 

本文介绍了一种针对MobileViT模型中多头自注意力(MHA)效率瓶颈的解决方案——可分离自注意力MobileViT v2。该方法具有线性复杂度的可分离自注意方法，并且使用基于元素的操作来计算自注意力，从而使其在资源受限的设备上的执行效率更高。

MobileViTv2在多个移动视觉任务上都是最先进的，包括ImageNet对象分类和MS-COCO对象检测。通过大约300万个参数，MobileViTv2在ImageNet数据集上获得了75.6%的top-1精度，比MobileViT高出约1%，同时在移动设备上运行速度快3.2倍。

![](https://i-blog.csdnimg.cn/blog_migrate/9ddf5b16b6724217d1c7a8079bcf96f9.png)

### 1.2 网络结构 

降低多头自注意时间复杂度有两个方向：

 *  tokens：在自注意层引入sparsity，在输入序列中每个token引入tokens一个子集；使用预定义模式限制token输入（不接受所有的tokens而是接受子集，缺点训练样本少性能下降很快）或者使用局部敏感的hash分组tokens（大型序列上才能看到提升）
 *  patches：通过低秩矩阵估计得到近似自注意矩阵，由线性连接将自注意操作分解成多个更小的自注意操作（Linformer使用batch-wise矩阵乘法）

本文主要是为了解决v1版本的高延迟问题：

1.  用分离自注意代替多头自注意提高效率
2.  使用element-wise操作替代batch-wise矩阵乘法

![](https://i-blog.csdnimg.cn/blog_migrate/f3ea315428848a61e598f0aa5c47e25c.png)

 *  (a)是一种标准的多头自注意(MHA)变压器。
 *  (b)在(a)中通过引入token投影层扩展MHA，将k个token投影到预定义数量的token p，从而将复杂度从O(k2)降低到O(k)。然而，它仍然使用昂贵的操作(例如，批量矩阵乘法)来计算自我注意，影响资源受限设备上的延迟。
 *  (c)是提出的可分离的自我注意层，其复杂性是线性的即O(k)，并使用元素操作来更快的推断。  
    

#### MHA 

dh=d/h,最后输出k个d维tokens，这个输出会在做一次矩阵乘法变成k\*d维向量，作为最后的输出。

![](https://i-blog.csdnimg.cn/blog_migrate/e69f15280594aa41ce44e18fb6f69426.png)

#### Separable self-attention 

论文中提到的解决方案的关键是将自我关注计算分成两个线性计算。

具体来说，该方法使用一个潜在标记来计算上下文分数，然后使用这些分数来重新加权输入标记，生成一个上下文向量。

由于自我关注计算与潜在标记的计算相关，因此这种方法可以将自我关注计算的复杂性从 O(k^2) 降低到 O(k)，其中 k 是标记的数量。

![](https://i-blog.csdnimg.cn/blog_migrate/6a89285051a6028d7018e54c36694c9b.png)

 *  分支L：用矩阵(b)L将x中每个d维向量映射到标量，计算(b)L与x的距离得到一个k维向量，这个k维向量softmax后就是上下文得分cs；
 *  分支K：直接矩阵相乘得到输出Xk,与cs相乘并相加k层，得到cv，cv类似于MHA的a矩阵，也编码了所有x的输入；
 *  分支V：线性映射并由ReLU激活得到Xv，然后与cv element-wise相乘，最后通过线性层得到最后的输出。

### 1.3 实验 

数据集

论文中使用了多个数据集进行实验，包括ImageNet-1k、ImageNet-21k-P和MS-COCO。

 *  在ImageNet-1k数据集上，作者使用了一个训练集和一个验证集，训练集包含128万张图像，验证集包含5万张图像。
 *  在ImageNet-21k-P数据集上，作者使用了一个训练集和一个验证集，训练集包含1100万张图像，验证集包含52万张图像。
 *  在MS-COCO数据集上，作者使用了一个训练集和一个验证集，训练集包含8万张图像，验证集包含4千张图像。

训练方法

 *  在ImageNet-1k数据集上，作者使用了AdamW算法进行训练，使用了一个有效的批次大小为1024的图像（128个图像每个GPU×8个GPU），训练300个时期。
 *  在ImageNet-21k-P数据集上，作者使用了与ImageNet-1k数据集相同的训练方法，但是使用了一个有效的批次大小为4096的图像（64个图像每个GPU×64个GPU），训练80个时期。
 *  在MS-COCO数据集上，作者使用了与ImageNet-1k数据集相同的训练方法，但是使用了一个有效的批次大小为128的图像。

（1）与自我注意方法的比较

![](https://i-blog.csdnimg.cn/blog_migrate/710f97d09376559fd0f89ec72bcbecbe.png)

 （2）ImageNet-1k验证集上的分类性能

![](https://i-blog.csdnimg.cn/blog_migrate/14621052da6636b028a9b4f4d844c898.png)

（3）ADE20k 和 PASCAL VOC 2012 数据集上的语义分割结果

![](https://i-blog.csdnimg.cn/blog_migrate/a7c3438549c956bca027c0a5728c6316.png)

（4）在MS-COCO数据集上使用SSDLite进行对象检测

![](https://i-blog.csdnimg.cn/blog_migrate/35c0dde68ee176991cdf0673bfdf85fc.png)

（5）MobileViTv2 模型不同输出步幅 (OS) 的上下文分数图

![](https://i-blog.csdnimg.cn/blog_migrate/72ee528bff1a6b7b048441350263f94b.png)

## 🚀二、具体添加方法 

#### 第①步：在common.py中添加MobileViTv2模块 

首先，定义卷积层。

分为普通卷积层和深度可分离卷积层

```java
def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 普通卷积层
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# 深度可分离卷积
class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# 带bn的1×1卷积分支
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
```

接着，构造ViT模块。

Transformer Encoder模块中编码

```java
#---ViT部分---#

# 规范化层的类封装
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
 '''
  dim: 输入和输出维度
  fn: 前馈网络层，选择Multi-Head Attn和MLP二者之一
 '''
        super().__init__()
        # LayerNorm: ( a - mean(last 2 dim) ) / sqrt( var(last 2 dim) )
        # 数据归一化的输入维度设定，以及保存前馈层

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
'''
 dim： 输入和输出维度
 hidden_dim：  中间层的维度
 dropout：  dropout操作的概率参数p
'''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        # 表示1/(sqrt(dim_head))用于消除误差，保证方差为1，避免向量内积过大导致的softmax将许多输出置0的情况
        # 可以看原文《attention is all you need》中关于Scale Dot-Product Attention如何抑制内积过大
        self.scale = dim_head ** -0.5
        # dim =  > 0 时，表示mask第d维度，对相同的第d维度，进行softmax
        # dim =  < 0 时，表示mask倒数第d维度，对相同的倒数第d维度，进行softmax
        self.attend = nn.Softmax(dim = -1)
        # 生成qkv矩阵，三个矩阵被放在一起，后续会被分开
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 如果是多头注意力机制则需要进行全连接和防止过拟合，否则输出不做更改
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        # 分割成q、k、v三个矩阵
        # qkv为 inner_dim * 3，其中inner_dim = heads * dim_head
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # qkv的维度是(3, inner_dim = heads * dim_head)
        # 'b n (h d) -> b h n d' 重新按思路分离出8个头，一共8组q,k,v矩阵
        # rearrange后维度变成 (3, heads, dim, dim_head)
        # 经过map后，q、k、v维度变成(1, heads, dim, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # query * key 得到对value的注意力预测，并通过向量内积缩放防止softmax无效化部分参数
        # heads * dim * dim
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 对最后一个维度进行softmax后得到预测的概率值
        attn = self.attend(dots)
        # 乘积得到预测结果
        # out -> heads * dim * dim_head
        out = torch.matmul(attn, v)
        # 重组张量，将heads维度重新还原
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# Transformer模块编码
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
```

 然后，MV2模块

![](https://i-blog.csdnimg.cn/blog_migrate/0bb18c624fc9aa944e73c57a2fc5b41c.png)

分文stride=1和stride=2两种。

```java
# MV2模块
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1: # 扩张率
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),# 3×3的卷积层
                nn.BatchNorm2d(hidden_dim), # BN层
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(oup), # BN层
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(hidden_dim), # BN层 
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(hidden_dim),# BN层 
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(oup), # BN层
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

最后，核心模块 MobileViT\_Block

![](https://i-blog.csdnimg.cn/blog_migrate/a8c0d0c46fabfd711e7ddc956b99037a.png)

分为局部表征模块和全局表征模块。

```java
# MobileViTv2_Block模块（核心部分）
class MobileViTv2_Block(nn.Module):
    def __init__(self, sim_channel, dim=64, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64 * 2), dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size # 获取h和w
        self.dwc = DWConv(sim_channel, sim_channel, kernel_size) # 3×3可分离卷积
        self.conv2 = conv_1x1_bn(sim_channel, dim) # 1×1的卷积层
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout) # Transformer进行编码操作
        self.conv3 = conv_1x1_bn(dim, sim_channel) # 1×1的卷积层
        self.mv2 = MV2Block(sim_channel, sim_channel) # MV2模块

    def forward(self, x):
        # Local representations #mg
        x = self.dwc(x)
        x = self.conv2(x)
        # Global representations #mg
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        x = self.conv3(x)
        x = self.mv2(x)
        return x
```

以下是完整代码：

将以下代码复制粘贴到common.py文件的末尾

```java
# MobileViTv2
from einops import rearrange
import math


def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# 普通卷积层
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# 深度可分离卷积
class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# 带bn的1×1卷积分支
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
#---ViT部分---#

# 规范化层的类封装
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
 '''
  dim: 输入和输出维度
  fn: 前馈网络层，选择Multi-Head Attn和MLP二者之一
 '''
        super().__init__()
        # LayerNorm: ( a - mean(last 2 dim) ) / sqrt( var(last 2 dim) )
        # 数据归一化的输入维度设定，以及保存前馈层

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
'''
 dim： 输入和输出维度
 hidden_dim：  中间层的维度
 dropout：  dropout操作的概率参数p
'''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)
 
        self.heads = heads
        # 表示1/(sqrt(dim_head))用于消除误差，保证方差为1，避免向量内积过大导致的softmax将许多输出置0的情况
        # 可以看原文《attention is all you need》中关于Scale Dot-Product Attention如何抑制内积过大
        self.scale = dim_head ** -0.5
        # dim =  > 0 时，表示mask第d维度，对相同的第d维度，进行softmax
        # dim =  < 0 时，表示mask倒数第d维度，对相同的倒数第d维度，进行softmax
        self.attend = nn.Softmax(dim = -1)
        # 生成qkv矩阵，三个矩阵被放在一起，后续会被分开
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 如果是多头注意力机制则需要进行全连接和防止过拟合，否则输出不做更改
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
    def forward(self, x):
        # 分割成q、k、v三个矩阵
        # qkv为 inner_dim * 3，其中inner_dim = heads * dim_head
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # qkv的维度是(3, inner_dim = heads * dim_head)
        # 'b n (h d) -> b h n d' 重新按思路分离出8个头，一共8组q,k,v矩阵
        # rearrange后维度变成 (3, heads, dim, dim_head)
        # 经过map后，q、k、v维度变成(1, heads, dim, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # query * key 得到对value的注意力预测，并通过向量内积缩放防止softmax无效化部分参数
        # heads * dim * dim
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 对最后一个维度进行softmax后得到预测的概率值
        attn = self.attend(dots)
        # 乘积得到预测结果
        # out -> heads * dim * dim_head
        out = torch.matmul(attn, v)
        # 重组张量，将heads维度重新还原
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer模块编码
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ---MobileViTv2部分--- #
# MV2模块
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1: # 扩张率
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),# 3×3的卷积层
                nn.BatchNorm2d(hidden_dim), # BN层
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(oup), # BN层
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(hidden_dim), # BN层 
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(hidden_dim),# BN层 
                nn.SiLU(), # SiLU函数
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), # 1×1的卷积层
                nn.BatchNorm2d(oup), # BN层
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileViTv2_Block模块（核心部分）
class MobileViTv2_Block(nn.Module):
    def __init__(self, sim_channel, dim=64, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64 * 2), dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size # 获取h和w
        self.dwc = DWConv(sim_channel, sim_channel, kernel_size) # 3×3可分离卷积
        self.conv2 = conv_1x1_bn(sim_channel, dim) # 1×1的卷积层
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout) # Transformer进行编码操作
        self.conv3 = conv_1x1_bn(dim, sim_channel) # 1×1的卷积层
        self.mv2 = MV2Block(sim_channel, sim_channel) # MV2模块

    def forward(self, x):
        # Local representations #mg
        x = self.dwc(x)
        x = self.conv2(x)
        # Global representations #mg
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        x = self.conv3(x)
        x = self.mv2(x)
        return x
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/e8a4667c6758e5693f16ad92ae54486d.png)

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat: 语句，在其后面加上下面代码：

```java
# mobilevit v2
        elif m in [MobileViTv2_Block]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2]
            if m in [MobileViTv2_Block]:
                args.insert(2, n)
                n = 1
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/75288093ab35e5cefe3de496711138ab.png)

#### 第③步：创建自定义的yaml文件 

yaml文件配置完整代码如下：

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
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, MobileViTv2_Block, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)


   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步 验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/6e8e2e4c6f9cf0320963218bb1ca4a28.png)

这样就OK啦~

> 代码参考：
> 
> [YOLOv5、YOLOv8改进MobileViTv2主干系列：全网首发最新苹果续作加强版 MobileViTv2结构（二），提出移动视觉 Transformer 的可分离自注意力机制，高效涨点提速\_mobilevitv2原论文\_芒果汁没有芒果的博客-CSDN博客][YOLOv5_YOLOv8_MobileViTv2_ MobileViTv2_ Transformer _mobilevitv2_-CSDN]

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

![](https://i-blog.csdnimg.cn/blog_migrate/25caddd9f389306104be0d196cefa0b6.gif)


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
[YOLOv5_20_BiFormer_CVPR2023]: https://blog.csdn.net/weixin_43334693/article/details/132203200?spm=1001.2014.3001.5502
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132367429%22%2C%22source%22%3A%22weixin_43334693%22%7D
[MobileViT v2_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81MobileViT%20v2%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0%C2%A0
[1.1 _]: #1.1%20%E7%AE%80%E4%BB%8B%C2%A0
[1.2]: #1.2%20%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[MHA]: #MHA
[Separable self-attention]: #Separable%20self-attention
[1.3]: #1.3%20%E5%AE%9E%E9%AA%8C
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95%C2%A0
[common.py_MobileViTv2]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%20%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2110.02178]: https://arxiv.org/abs/2110.02178
[GitHub - apple_ml-cvnets_ CVNets_ A library for training computer vision networks]: https://github.com/apple/ml-cvnets
[YOLOv5_YOLOv8_MobileViTv2_ MobileViTv2_ Transformer _mobilevitv2_-CSDN]: https://yoloair.blog.csdn.net/article/details/127107723
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