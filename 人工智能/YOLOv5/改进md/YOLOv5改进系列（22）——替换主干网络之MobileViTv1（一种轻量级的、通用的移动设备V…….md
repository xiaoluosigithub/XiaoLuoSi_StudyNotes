![](https://i-blog.csdnimg.cn/blog_migrate/a9c7780f2200c8ad05f660b0f4738abc.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ddeae90d306eef59f9100dbb897339df.png)

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
[  
YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]

[YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）][YOLOv5_20_BiFormer_CVPR2023]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

![](https://i-blog.csdnimg.cn/blog_migrate/b432bc3c0686ff75fa31365d201e1c46.gif)

目录

[🚀一、MobileViT v1介绍 ][MobileViT v1_]

[1.1 简介][1.1]

[1.2 网络结构][1.2]

[（1）MV2 ][1_MV2]

[（2）MobileViTblock][2_MobileViTblock]

[1.3 实验][1.3]

[（1）和CNN对比][1_CNN]

[（2）和ViT对比][2_ViT]

[（3）移动端目标检测 ][3_]

[（4）移动端实例分割 ][4_]

[（5）移动设备的性能 ][5_]

[🚀二、具体添加方法 ][Link 1]

[第①步：在common.py中添加MobileViTv1模块][common.py_MobileViTv1]

[第②步：修改yolo.py文件][yolo.py]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步 验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/4b78b961ca9b5ef209b81702938f7f5b.gif)

## 🚀一、MobileViT v1介绍 

>  *  论文题目：《MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer》
>  *  论文地址：[https://arxiv.org/abs/2110.02178][https_arxiv.org_abs_2110.02178]
>  *  源码地址：[GitHub - apple/ml-cvnets: CVNets: A library for training computer vision networks][GitHub - apple_ml-cvnets_ CVNets_ A library for training computer vision networks]

![](https://i-blog.csdnimg.cn/blog_migrate/c6a0614544b6cdc5d5c7e924c8e70ae7.png)

（Apple(⊙o⊙) ）

### 1.1 简介 

MobileViT网络是由苹果公司提出了一种轻量级的、通用的移动设备 vision transformer，将CNN和ViT的优势相结合，提高了在移动视觉任务中的性能。

以往的研究主要集中在轻量级卷积神经网络和自注意力ViTs，其中CNN具有局部感知性，参数较少，ViTs具有全局感知性，但参数较多。然而，这些方法在移动视觉任务中存在一些问题，如性能不够理想、延迟较高等。

本篇论文提出了MobileViT的研究方法，将transformers作为卷积的方式进行全局信息处理，实现了轻量级和低延迟的移动视觉任务网络。

研究结果表明，MobileViT在不同任务和数据集上明显优于基于CNN和ViT的网络。在ImageNet-1k数据集上取得了最佳结果。

### 1.2 网络结构 

![](https://i-blog.csdnimg.cn/blog_migrate/8767f5f00610877f8281ae0ef63af1b0.png)

上面那个图展示就是标准视觉ViT模型，下面就是今天要介绍的MobileViT的网络结构。

`主要由`MV2和MobileViTblock两个模块组成，下面我们来介绍下这两个模块：

#### （1）MV2 

MV2就是MobileNet v2（直通车：[【轻量化网络系列（2）】MobileNetV2论文超详细解读（翻译 ＋学习笔记+代码实现）][2_MobileNetV2_]）里面Inverted Residual Block，即下面的图所示的结构。

![](https://i-blog.csdnimg.cn/blog_migrate/f28b99a7181fae3b313a2391fa8b944c.png)

图中MV2是当stride等于1时的MV2结构，上图中标有向下箭头的MV2结构代表stride等于2的情况，即需要进行下采样。

#### （2）MobileViTblock 

![](https://i-blog.csdnimg.cn/blog_migrate/04a615e548c2abb5a37800acafef68ea.png)

首先将特征图通过一个卷积层，卷积核大小为n×n，然后再通过一个卷积核大小为1×1的卷积层进行通道调整。

接着依次通过Unfold、Transformer、Fold结构进行全局特征建模，然后再通过一个卷积核大小为1×1的卷积层将通道调整为原始大小。

接着通过shortcut捷径分支与原始输入特征图按通道concat拼接。

最后再通过一个卷积核大小为n×n的卷积层进行特征融合得到最终的输出。

### 1.3 实验 

#### （1）和CNN对比 

![](https://i-blog.csdnimg.cn/blog_migrate/d15e87ac6eeaf39f50980cf312b18816.png)

#### （2）和ViT对比 

![](https://i-blog.csdnimg.cn/blog_migrate/2b13a52d99ae281a449afe4765bd35e7.png)

#### （3）移动端目标检测 

![](https://i-blog.csdnimg.cn/blog_migrate/c201dc3e04ef8dc6508ce18a4289f08b.png)

#### （4）移动端实例分割 

![](https://i-blog.csdnimg.cn/blog_migrate/08de1f103b98bdba80bbac272d50f8c9.png)

#### （5）移动设备的性能 

![](https://i-blog.csdnimg.cn/blog_migrate/bc294fca99da0644e0a7c57555300b1b.png)

## 🚀二、具体添加方法 

#### 第①步：在common.py中添加MobileViTv1模块 

将以下代码复制粘贴到common.py文件的末尾

```java
from einops import rearrange

class TAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class MoblieTrans(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, TAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2B(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(ch_in * expansion)
        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViT_Block(nn.Module):
    def __init__(self, ch_in, dim=64, depth=2, kernel_size=3, patch_size=(2, 2), mlp_dim=int(64 * 2), dropout=0.):
        super().__init__()  # mg
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(ch_in, ch_in, kernel_size)
        self.conv2 = conv_1x1_bn(ch_in, dim)

        self.transformer = MoblieTrans(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, ch_in)
        self.conv4 = conv_nxn_bn(2 * ch_in, ch_in, kernel_size)

    def forward(self, x):
        y = x.clone()  # torch.Size for
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)  # torch.Size for

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


def conv_1x1_bn(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.SiLU()
    )


def conv_nxn_bn(ch_in, ch_out, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
```

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat:语句，在其后面加上下面代码：

```java
# mobilevit v1
        elif m in [MobileViT_Block]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not outputss
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/642347fa295a0aa5e88241ac2a9d9419.png)

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
   [-1, 3, MobileViT_Block, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)


   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步 验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/96c3df7f99e9396add4c155eed86cb2b.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/c942408284af6cfc00393cd42f436339.gif)


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
[MobileViT v1_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81MobileViT%20v1%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[1.2]: #1.2%20%E6%96%B9%E6%B3%95
[1_MV2]: #%EF%BC%881%EF%BC%89MV2%C2%A0
[2_MobileViTblock]: #%EF%BC%882%EF%BC%89MobileViTblock
[1.3]: #1.3%20%E5%AE%9E%E9%AA%8C
[1_CNN]: #%EF%BC%881%EF%BC%89%E5%92%8CCNN%E5%AF%B9%E6%AF%94
[2_ViT]: #%EF%BC%882%EF%BC%89%E5%92%8CViT%E5%AF%B9%E6%AF%94
[3_]: #%EF%BC%883%EF%BC%89%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%C2%A0
[4_]: #%EF%BC%884%EF%BC%89%E7%A7%BB%E5%8A%A8%E7%AB%AF%E5%AE%9E%E4%BE%8B%E5%88%86%E5%89%B2%C2%A0
[5_]: #%EF%BC%885%EF%BC%89%E7%A7%BB%E5%8A%A8%E8%AE%BE%E5%A4%87%E7%9A%84%E6%80%A7%E8%83%BD%C2%A0
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95%C2%A0
[common.py_MobileViTv1]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%20%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2110.02178]: https://arxiv.org/abs/2110.02178
[GitHub - apple_ml-cvnets_ CVNets_ A library for training computer vision networks]: https://github.com/apple/ml-cvnets
[2_MobileNetV2_]: https://blog.csdn.net/weixin_43334693/article/details/130772823?spm=1001.2014.3001.5501
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