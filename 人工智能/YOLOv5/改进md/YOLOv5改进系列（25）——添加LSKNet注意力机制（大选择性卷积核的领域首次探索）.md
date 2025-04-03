![](https://i-blog.csdnimg.cn/blog_migrate/5e317ebd10ab1b43271312b2285be4a4.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/a54e3ce6fce188752e7c246cd35259b7.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/a59ac642a71c9e7c575dddc3fcd95d22.gif)

目录

[🚀 一、LSKNet介绍][_LSKNet]

[1.1 LSKNet简介][1.1 LSKNet]

[1.2 LSKNet网络结构][1.2 LSKNet]

[🚀二、LSKNet源码讲解 ][LSKNet_]

[🚀三、具体添加方法][Link 1]

[ 3.1 添加顺序 ][3.1 _]

[3.2 具体添加步骤 ][3.2 _]

[第①步：在common.py中添加LSK模块 ][common.py_LSK_]

[第②步：修改yolo.py文件][yolo.py]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/a56b1cc7b8b9dbf25447897d7efdb592.gif)

## 🚀 一、LSKNet介绍 

>  *  论文题目：《Large Selective Kernel Network for Remote Sensing Object Detection》
>  *  论文地址：[https://arxiv.org/pdf/2303.09030.pdf][https_arxiv.org_pdf_2303.09030.pdf]
>  *  代码实现： [GitHub - zcablii/LSKNet: (ICCV 2023) Large Selective Kernel Network for Remote Sensing Object Detection][GitHub - zcablii_LSKNet_ _ICCV 2023_ Large Selective Kernel Network for Remote Sensing Object Detection]

### 1.1 LSKNet简介 

LSKNet是指Large Selective Kernel Network的缩写，翻译为大型选择性核网络。这是南开大学在ICCV2023会议上新提出的目标检测旋转算法，基本原理就是通过一系列Depth-wise 卷积核和空间选择机制来动态调整目标的感受野，从而允许模型适应不同背景的目标检测。

这篇论文着重于遥感目标的特征提取，原始遥感目标检测的不足：

1.  精确识别遥感影像中的目标通常需要充分的背景信息支持。有限的背景区域可能会对模型的识别效果产生影响，例如在背景信息稀缺的情况下，模型可能会错误地将十字路口误认为是道路。
2.  不同类型的目标对背景信息的需求是多样的。以足球场为例，其可通过明显的球场边界线进行区分，因此所需的背景信息较少。然而，十字路口与道路相似，容易受到树木和其他遮挡物的影响，因此为了进行准确的识别，需要提供足够广泛的背景信息范围。

![](https://i-blog.csdnimg.cn/blog_migrate/ae7d7de564135258c21d71483687181a.png)

经验证，LSKNet虽然结构简单，但能够获得优异的检测性能，在HRSC2016、DOTA-v1.0、FAIR1M-v1.0三个典型数据集上都取得了SOTA。

### 1.2 LSKNet网络结构 

![](https://i-blog.csdnimg.cn/blog_migrate/e655348efb4dd8bce37426fef01c1ab3.png)

本文改进方法： 

1.  通过使用卷积核大小为5\*5的普通卷积与卷积核大小为7\*7的膨胀卷积，相结合来达到卷积核大小为23\*23的大核卷积的效果，从而使网络在进行特征提取的时候获得超大的感受野，进而，获得丰富的上下文信息，也就是背景信息。这一点是在目前较为流行的主干特征提取网络（resnet系列，swin-transform）中是没有的。
2.  通过使用普通二维卷积与Sigmoid函数设计了一种模块选择网络，使得网络可以针对不同的检测目标动态的选取不同感受野大小的卷积，从而获得最好的检测效果。例如，针对自身特征就比较明显的对象（足球场）就选择感受野较小的卷积来提取特征。针对自身特征不太明显的对象（船只，交叉路口）则选取感觉野较大的卷积来提取特征，利用目标周围的背景信息来提高这类目标识别的准确率。

具体方法：

（1）首先，LSKNet通过使用普通卷积和膨胀卷积，分别生成两个不同的特征图。

（2）接着，通过应用卷积核大小为1\*1的卷积，将两者的通道数调整为相同的大小。再将它们叠加在一起，形成对应的特征图。

（3）然后，对该特征图进行平均池化和最大池化操作，将两者堆叠在一起。通过卷积和Sigmoid操作，获得针对不同卷积核大小的选择权重。

（4）最后，将得到的权重与之前提到的特征图相乘和相加，再与最初的输入X相乘，从而得到最终的输出Y。

## 🚀二、LSKNet源码讲解 

LSKNet的代码如下： 

```java
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

# 1. Mlp 模块 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
# 2. LSKblock 模块 
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
 
    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
 
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
 
 
# 3. Attention 模块
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
 
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
 
    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
 
# 4. Block 模块 
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
 
    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
 
# 5. OverlapPatchEmbed 模块  
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
 
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)
 
 
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

# 6. LSKNet 模块   
@ROTATED_BACKBONES.register_module()
class LSKNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
 
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)
 
            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
 
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
 
 
    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet, self).init_weights()
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False
 
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better
 
    def get_classifier(self):
        return self.head
 
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
 
    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs
 
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x
 
# 7. DWConv 模块 
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
 
    def forward(self, x):
        x = self.dwconv(x)
        return x
 
# 8. _conv_filter 函数  
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
 
    return out_dict
```

一共有七个模块和一个函数：

1.  Mlp 模块
2.  LSKblock 模块
3.  Attention 模块
4.  Block 模块
5.  OverlapPatchEmbed 模块
6.  LSKNet 模块
7.  DWConv 模块
8.  \_conv\_filter 函数

这里我们重点讲一下 LSKblock 模块：

![](https://i-blog.csdnimg.cn/blog_migrate/3be448a3aed9a8b186ffdc24848d3afd.png)

主要包括了：

1.  self.conv0： 一个5x5的卷积层，使用depthwise卷积（groups=dim），用于从输入特征图中提取信息。
2.  self.conv\_spatial: 一个7x7的卷积层，用于处理输入特征图的空间信息，具有较大的padding和dilation，dilation`=3` 表示膨胀卷积。
3.  self.conv1和self.conv2： 一个1x1的卷积层，分别用于处理conv0和conv\_spatial的输出，将通道数减半。
4.  self.conv\_squeeze： 一个7x7的卷积层，用于对avg\_attn和max\_attn进行通道方向的融合，通过sigmoid激活函数得到注意力权重。
5.  self.conv： 一个1x1的卷积层，用于将融合后的注意力权重与输入特征图进行融合。

## 🚀三、具体添加方法 

### 3.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py  -->  设定网络结构的传参细节，将LSKblock类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml  --> 新建一个文件夹，如yolov5s\_LSK.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py  --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 3.2 具体添加步骤 

#### 第①步：在common.py中添加LSK模块 

将下面的LSK代码复制粘贴到common.py文件的末尾

```java
# LSKNet
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 一个5x5的卷积层，groups=dim 表示使用分组卷积，其中 dim 是输入通道数。
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 一个7x7的卷积层，使用了步幅1、padding为9、groups=dim 表示分组卷积、dilation=3 表示膨胀卷积。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 一个1x1的卷积层，将 dim 通道减半。
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        # 一个1x1的卷积层，将 dim 通道减半。
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 一个7x7的卷积层，padding为3，用于对两个路径的信息进行压缩。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 一个1x1的卷积层，将 dim//2 通道的信息还原为 dim。
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # 使用 self.conv0 和 self.conv_spatial 对输入特征图进行不同的卷积操作，得到 attn1 和 attn2。
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        # 使用 self.conv1 和 self.conv2 对 attn1 和 attn2 进行进一步的卷积操作。
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 将 attn1 和 attn2 沿着通道维度拼接在一起，得到 attn。
        attn = torch.cat([attn1, attn2], dim=1)
        # 计算 attn 在通道维度上的平均值和最大值，得到 avg_attn 和 max_attn。
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 将 avg_attn 和 max_attn 沿着通道维度拼接在一起，得到 agg。
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 对 agg 进行压缩，通过 self.conv_squeeze 进行sigmoid激活，得到注意力权重 sig。
        sig = self.conv_squeeze(agg).sigmoid()
        # 使用 sig 对 attn1 和 attn2 进行加权融合，得到最终的注意力结果 attn。
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 将 attn 经过 self.conv 进行进一步处理，得到最终的输出，返回 x * attn。
        attn = self.conv(attn)
        return x * attn
```

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat: 语句，在其后面加上下面代码：

```java
elif m is LSKblock:
            c1 = ch[f]
            args = [c1, *args[0:]]
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/bcadbf850b53f6e0442a6d9ecdcc6a23.png)

#### 第③步：创建自定义的yaml文件 

yaml文件配置完整代码如下：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 1  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.1 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],    # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],    # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],    # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],   # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, LSKblock, []],   
   [-1, 1, SPPF, [1024, 5]],      # 10
  ]

# YOLOv5 v6.1 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 14

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 18 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 15], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],   # 21 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 11], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 24 (P5/32-large)

   [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/86cc9a5df26cb6952fcf5a1488176668.png)

这样就OK啦！

> 代码参考：
> 
> [改进YOLO系列 | YOLOv5/v7 引入选择性注意力 LSK 模块 | 《ICCV 2023 Large Selective Kernel Network》][YOLO_ _ YOLOv5_v7 _ LSK _ _ _ICCV 2023 Large Selective Kernel Network]

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

![](https://i-blog.csdnimg.cn/blog_migrate/11e2b3578c6fa82bf0ee821b7ecebf49.gif)


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
[YOLOv5_24_MobileViTv3]: https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502
[_LSKNet]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81LSKNet%E4%BB%8B%E7%BB%8D
[1.1 LSKNet]: #1.1%C2%A0LSKNet%E7%AE%80%E4%BB%8B
[1.2 LSKNet]: #1.2%C2%A0LSKNet%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[LSKNet_]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81LSKNet%E6%BA%90%E7%A0%81%E8%AE%B2%E8%A7%A3%C2%A0
[Link 1]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95
[3.1 _]: #%C2%A03.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[3.2 _]: #3.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_LSK_]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[yolo.py]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 2]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_pdf_2303.09030.pdf]: https://arxiv.org/pdf/2303.09030.pdf
[GitHub - zcablii_LSKNet_ _ICCV 2023_ Large Selective Kernel Network for Remote Sensing Object Detection]: https://github.com/zcablii/LSKNet
[YOLO_ _ YOLOv5_v7 _ LSK _ _ _ICCV 2023 Large Selective Kernel Network]: https://yolov5.blog.csdn.net/article/details/132040748
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