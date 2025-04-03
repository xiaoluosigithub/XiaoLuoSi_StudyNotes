![](https://i-blog.csdnimg.cn/blog_migrate/7f30cb441e3fbc6c5aa1e2886fc1d9ce.gif)![](https://i-blog.csdnimg.cn/blog_migrate/75a213d9e1e88088f54fe797f262764b.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/86839dfe5fe7e59488cf10cc3d4041e3.gif)

目录

[🚀一、MobileViT v3介绍 ][MobileViT v3_]

[🚀二、具体添加方法 ][Link 1]

[第①步：在common.py中添加MobileViT v3模块][common.py_MobileViT v3]

[第②步：修改yolo.py文件][yolo.py]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步 验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/3709898b1adfc5eb46af11a6ed72c8ce.gif)

## 🚀一、MobileViT v3介绍 

>  *  论文题目：《MOBILEVITV3: MOBILE-FRIENDLY VISION TRANS-  
>     FORMER WITH SIMPLE AND EFFECTIVE FUSION OF LOCAL, GLOBAL AND INPUT FEATURES》
>  *  论文原文： [https://arxiv.org/abs/2209.15159][https_arxiv.org_abs_2209.15159]
>  *  源码地址：[https://github.com/micronDLA/MobileViTv3][https_github.com_micronDLA_MobileViTv3]
>  *  论文精读：[MobileViT v3论文超详细解读（翻译＋精读）\_路人贾'ω'的博客-CSDN博客][MobileViT v3_-CSDN]

 在之前的研究中，CNN模型足够轻量化但是精准度有待提高，ViT模型具有较好的识别能力但是模型参数量大，计算复杂，都不能满足移动端实时高效检测的需求。

MobileViT模型是2021年苹果公司提出的基于轻量化的ViT模型，该模型既具备ViT模型准确检测的优越性能，也具备CNN模型的轻量化优点，能极大程度上减少模型参数，对移动端友好，具备部署于移动设备的可能性。

 MobileViT v3是该公司2022年9月推出的第3个版本，该模型相较于初始版本有以下四个改进：

 *  首先，将3×3卷积层替换为1×1卷积层；
 *  第二，将局部表示块和全局表示块的特征融合在一起；
 *  第三，在生成MobileViT Block输出之前，在融合块中添加输入特征作为最后一步；
 *  最后，在局部表示块中，将普通的3×3卷积层替换为深度3×3卷积层。

MobileViTv3网络结构图如下所示：

![](https://i-blog.csdnimg.cn/blog_migrate/fd544d151fffbf3e5776ac55923b2a3f.png)

模型预测处理过程如下：

（1）将输入图像连接3×3标准卷积并做2倍下采样；之后通过5个MV2模块（如图（a）所示），其中步长为1的MV2模块进行特征提取，步长为2的MV2模块做2倍下采样；

（2）将得到的特征图间隔传入MobileViTV3 Block（如图（b）所示）和步长为2的MV2模块；

（3）接着使用3×3标准卷积进行通道压缩；

（4）最后进行全局平均池化来获取预测结果 。

MobileViT v3 Block模块是MobileViT v3核心部分，由局部表征模块、全局表征模块、融合模块三部分组成，具体介绍如下：

（1）局部表征模块 

对于输入的特征图像![X\in R^{H\times W\times C}](https://latex.csdn.net/eq?X%5Cin%20R%5E%7BH%5Ctimes%20W%5Ctimes%20C%7D)（![H](https://latex.csdn.net/eq?H)、![W](https://latex.csdn.net/eq?W)为图像的高、宽；![C](https://latex.csdn.net/eq?C)为输入特征图像的通道） ，首先采用一个3×3的深度卷积层和1×1的卷积层得到输出![X_{L}\in R^{H\times W\times d}](https://latex.csdn.net/eq?X_%7BL%7D%5Cin%20R%5E%7BH%5Ctimes%20W%5Ctimes%20d%7D)（![d](https://latex.csdn.net/eq?d)为输出特征图像的通道）。经过局部表征模块可以将特征图像![X](https://latex.csdn.net/eq?X)的局部空间信息映射到特定的维度![d](https://latex.csdn.net/eq?d)中。

（2）全局表征模块

将局部表征模块中的输出![X_{L}](https://latex.csdn.net/eq?X_%7BL%7D)作为全局表征模块的输入，然后将![X_{L}](https://latex.csdn.net/eq?X_%7BL%7D)展平为无重叠的图片块![X_{U}\in R^{P\times N\times d}](https://latex.csdn.net/eq?X_%7BU%7D%5Cin%20R%5E%7BP%5Ctimes%20N%5Ctimes%20d%7D)（![P](https://latex.csdn.net/eq?P)为展平图片块的大小，![P=w\times h](https://latex.csdn.net/eq?P%3Dw%5Ctimes%20h)，![w](https://latex.csdn.net/eq?w)、![h](https://latex.csdn.net/eq?h)分别为展平图片块的宽、高，![N](https://latex.csdn.net/eq?N)为图片块的个数，![N=W\times H/P](https://latex.csdn.net/eq?N%3DW%5Ctimes%20H/P)）。随后，将每个图片块上相同位置上的像素![p\in \left \{ 1,...,P \right \}](https://latex.csdn.net/eq?p%5Cin%20%5Cleft%20%5C%7B%201%2C...%2CP%20%5Cright%20%5C%7D)送入Transformer模块中进行编码，得到![X_{G}\in R^{P\times N\times d}](https://latex.csdn.net/eq?X_%7BG%7D%5Cin%20R%5E%7BP%5Ctimes%20N%5Ctimes%20d%7D)。 为了避免图片块之间的信息丢失，在全局表征模块的最后会将编码后的所有图片块重组还原，得到![X_{F}\in R^{H\times W\times d}](https://latex.csdn.net/eq?X_%7BF%7D%5Cin%20R%5E%7BH%5Ctimes%20W%5Ctimes%20d%7D)，然后输出到下一个模块。

（3）融合模块

将![X_{F}](https://latex.csdn.net/eq?X_%7BF%7D)送入特征融合模块中，使其映射到一个低维空间，得到![X^{'}\in R^{H\times ×W\times ×C}](https://latex.csdn.net/eq?X%5E%7B%27%7D%5Cin%20R%5E%7BH%5Ctimes%20%D7W%5Ctimes%20%D7C%7D)。 然后通过拼接操作，将局部表征模块的输出![X_{L}](https://latex.csdn.net/eq?X_%7BL%7D)与![X^{'}](https://latex.csdn.net/eq?X%5E%7B%27%7D)拼接起来，得到![X^{''}\in R^{H\times ×W\times ×2C}](https://latex.csdn.net/eq?X%5E%7B%27%27%7D%5Cin%20R%5E%7BH%5Ctimes%20%D7W%5Ctimes%20%D72C%7D)。接着采用1×1的卷积层来融合局部特征![X^{'}](https://latex.csdn.net/eq?X%5E%7B%27%7D)与全局特征![X^{''}](https://latex.csdn.net/eq?X%5E%7B%27%27%7D)，得到![X^{*}\in R^{H\times ×W\times ×2C}](https://latex.csdn.net/eq?X%5E%7B*%7D%5Cin%20R%5E%7BH%5Ctimes%20%D7W%5Ctimes%20%D72C%7D)。最后把刚获取的![X^{*}](https://latex.csdn.net/eq?X%5E%7B*%7D)和原始![X](https://latex.csdn.net/eq?X)进行相加操作，得到输出![Y\in R^{H\times W\times C}](https://latex.csdn.net/eq?Y%5Cin%20R%5E%7BH%5Ctimes%20W%5Ctimes%20C%7D)。

> 该过程需要伪代码可以私聊我~

## 🚀二、具体添加方法 

#### 第①步：在common.py中添加MobileViT v3模块 

首先，定义卷积层。

分为1×1卷积层和n×n（n=3）卷积层

```java
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
```

接着，构造ViT模块。

Transformer Encoder模块中编码

```java
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # mg
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)# mg
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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

class MBTransformer(nn.Module):
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

 然后，MV2模块。

![](https://i-blog.csdnimg.cn/blog_migrate/0bb18c624fc9aa944e73c57a2fc5b41c.png)

分为stride=1和stride=2两种。

```java
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

最后，核心模块 MobileViTv3\_block。

介绍部分看上面就行~

```java
class MobileViTv3_block(nn.Module):
    def __init__(self, channel, dim, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64*2), dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.mv01 = MV2Block(channel, channel) 
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = MBTransformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        z = x.clone()
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, z), 1)
        x = self.conv4(x)
        x = x + y
        x = self.mv01(x)
        return x
```

以下是完整代码：

将以下代码复制粘贴到common.py文件的末尾

```java
from einops import rearrange


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn # mg
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)# mg
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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

class MBTransformer(nn.Module):
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

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTv3_block(nn.Module):
    def __init__(self, channel, dim, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64*2), dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.mv01 = MV2Block(channel, channel) 
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = MBTransformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        z = x.clone()
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, z), 1)
        x = self.conv4(x)
        x = x + y
        x = self.mv01(x)
        return x
```

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat: 语句，在其后面加上下面代码：

```java
elif m in [MobileViTv3_block]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2]
            if m in [MobileViTv3_block]:
                args.insert(2, n)  
                n = 1
```

#### 第③步：创建自定义的yaml文件 

yaml文件配置完整代码如下：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel iscyy multiple
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
   [-1, 6, MobileViTv3_block, [256]],
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
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步 验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/39c3d274fc83ad1f23ee4a192e48886b.png)

这样就OK啦~

> 代码参考： 
> 
> [YOLOv5、YOLOv8改进主干：全网首发最新 MobileViTv3 系列最强改进版本（三）｜轻量化Transformer视觉转换器，简单有效地融合了本地全局和输入特征，高效涨点\_yolov8轻量化改进\_芒果汁没有芒果的博客-CSDN博客][YOLOv5_YOLOv8_ MobileViTv3 _Transformer_yolov8_-CSDN]

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

![](https://i-blog.csdnimg.cn/blog_migrate/16dba0d75cda098194f5e0d3473cbff1.gif)


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
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502
[MobileViT v3_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81MobileViT%20v3%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0%C2%A0%C2%A0
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95%C2%A0
[common.py_MobileViT v3]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%20%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2209.15159]: https://arxiv.org/abs/2209.15159
[https_github.com_micronDLA_MobileViTv3]: https://github.com/micronDLA/MobileViTv3
[MobileViT v3_-CSDN]: https://blog.csdn.net/weixin_43334693/article/details/132742052?spm=1001.2014.3001.5502
[YOLOv5_YOLOv8_ MobileViTv3 _Transformer_yolov8_-CSDN]: https://yoloair.blog.csdn.net/article/details/127107758
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