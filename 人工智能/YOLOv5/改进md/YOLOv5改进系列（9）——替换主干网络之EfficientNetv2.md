![](https://i-blog.csdnimg.cn/blog_migrate/8c20122446ccb917f472a492460af930.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/c5bb370b879774bbeac8cd2101e60ba8.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/a6f349a1381f4bd18c69bf6ffd4f54b5.gif)

目录

[🚀 一、EfficientNet介绍][_EfficientNet]

[1.1 EfficientNet V1][]

[1.2 EfficientNet V2][]

[🚀 二、YOLOv5结合EfficientNetv2][_YOLOv5_EfficientNetv2]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤][2.2]

[第①步：在common.py中添加EfficientNetv2模块][common.py_EfficientNetv2]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数 ][train.py_ _--cfg_]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/41846690bc6c67f97290aecb3a578f2b.gif)

## 🚀 一、EfficientNet介绍 

在之前的一些论文中，有的会通过增加网络的width即增加卷积核的个数（增加特征矩阵的channels）来提升网络的性能如图(b)所示，有的会通过增加网络的深度即使用更多的层结构来提升网络的性能如图(c)所示，有的会通过增加输入网络的分辨率来提升网络的性能如图(d)所示。而在EfficientNet中会同时增加网络的width、网络的深度以及输入网络的分辨率来提升网络的性能如图(e)所示：

![](https://i-blog.csdnimg.cn/blog_migrate/dbca7685cf2c9b3a4a66a74a2267a728.png#pic_center)

### 1.1 EfficientNet V1 

> [【轻量化网络系列（6）】EfficientNetV1论文超详细解读（翻译 ＋学习笔记+代码实现）][6_EfficientNetV1_]

 简介

EfficientNetv1主要是用NAS（Neural Architecture Search）技术来搜索网络的图像输入分辨率r，网络的深度depth以及channel的宽度width三个参数的合理化配置。在之前的一些论文中，基本都是通过改变上述3个参数中的一个来提升网络的性能，而EfficientNetv1就是同时来探索这三个参数的影响。

创新点

（1）系统地研究了模型的缩放，发现平衡网络深度、宽度和分辨率可以带来更好的性能。

（2）提出了一种新的缩放方法，使用一个简单而高效的复合系数统一缩放深度/宽度/分辨率的所有维度。

（3）使用神经结构搜索来设计一个新的基线网络，并将其扩大到获得一个模型系列，称为EfficientNets。

网络模型结构

EfficientNet-B0 baseline 网络的结构配置如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/cc5f3d18ab97b1cff7b0414888c4fcac.png)

在 B0 中一共分为 9 个 stage，表中的卷积层后默认都跟有 BN 以及 Swish 激活函数。

stage 1 就是一个 3×3 的卷积层。stage 2 到 stage 8 就是在重复堆叠 MBConv。

MBConv 模块

![](https://i-blog.csdnimg.cn/blog_migrate/a05a41ec17b2788a7eaf02b64bade2b4.png)

（1）首先是一个 1×1 卷积用于升维，其输出 channel 是输入 channel 的 n 倍。

（2）紧接着通过一个 DW 卷积。

（3）然后通过一个 SE 模块，使用注意力机制调整特征矩阵。

（4）之后再通过 1×1 卷积进行降维。注意这里只有 BN，没有 swish 激活函数（其实就是对应线性激活函数）。

（5）最后跟一个dropout层。

### 1.2 EfficientNet V2 

> [【轻量化网络系列（7）】EfficientNetV2论文超详细解读（翻译 ＋学习笔记+代码实现）][7_EfficientNetV2_]

EfficientNet V1的不足

（1）输入分辨率大时训练比较慢

（2）深度depthwise卷积在网络浅层中比较慢

（3）用同样的缩放系数缩放网络的每个stage是次优的

简介

EfficientNetV2 主要使用训练感知神经结构搜索和缩放的组合；在EfficientNetV1的基础上，引入了Fused-MBConv到搜索空间中；引入渐进式学习策略、自适应正则强度调整机制使得训练更快；进一步关注模型的推理速度与训练速度。

创新点

（1）本文提出了EfficientNet V2，一个更小更快的模型，基于training-aware NAS和scaling，EfficientNetV2在训练速度和参数效率方面都优于之前的模型。

（2）本文提出了一种改进的渐进式训练方法，它自适应的调整正则化和输入大小，通过实验证明该方法既加快了训练速度，同时也提高了准确性。

网络模型结构

![](https://i-blog.csdnimg.cn/blog_migrate/d4b1722b5031b5a067a36e37cc2e7e20.png)

与V1的不同：

（1）除了使用 MBConv 之外还使用了 Fused-MBConv 模块，加快训练速度与提升性能

（2）使用较小的 expansion ratio (之前是6)，从而减少内存的访问量

（3）趋向于选择kernel大小为3的卷积核，但是会增加多个卷积用以提升感受野，（ V1 中有 5 × 5 )

（4）移除了最后一个stride为1的stage，从而减少部分参数和内存访问

## 🚀 二、YOLOv5结合EfficientNetv2 

### 2.1 添加顺序 

之前在讲添加注意力机制时我们就介绍过改进网络的顺序，替换主干网络也是大同小异的。

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将EfficientNetv2类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）

（3） models/yolov5\*.yaml -->  修改现有模型结构配置文件

 *  当引入新的层时，要修改后续的结构中的from参数
 *  当仅替换主千网络时，要注意特征图的变换，/8，/16，/32

（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加EfficientNetv2模块 

将以下代码复制粘贴到common.py文件的末尾

```java
#EfficientNetV2

class stem(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        super().__init__()
        # kernel_size为3时，padding 为1，kernel为1时，padding为0
        padding = (kernel_size - 1) // 2
        # 由于要加bn层，所以不加偏置
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize

    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SqueezeExcite_efficientv2(nn.Module):
    def __init__(self, c1, c2, se_ratio=0.25, act_layer=nn.ReLU):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        reduced_chs = int(c1 * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(c1, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, c2, 1, bias=True)

    def forward(self, x):
        # 先全局平均池化
        x_se = self.avg_pool(x)
        # 再全连接（这里是用的1x1卷积，效果与全连接一样，但速度快）
        x_se = self.conv_reduce(x_se)
        # ReLU激活
        x_se = self.act1(x_se)
        # 再全连接
        x_se = self.conv_expand(x_se)
        # sigmoid激活
        x_se = self.gate_fn(x_se)
        # 将x_se 维度扩展为和x一样的维度
        x = x * (x_se.expand_as(x))
        return x

# Fused-MBConv 将 MBConv 中的 depthwise conv3×3 和扩展 conv1×1 替换为单个常规 conv3×3。
class FusedMBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()
        # shorcut 是指到残差结构 expansion是为了先升维，再卷积，再降维，再残差
        self.has_shortcut = (s == 1 and c1 == c2)  # 只要是步长为1并且输入输出特征图大小相等，就是True 就可以使用到残差结构连接
        self.has_expansion = expansion != 1  # expansion==1 为false expansion不为1时，输出特征图维度就为expansion*c1，k倍的c1,扩展维度
        expanded_c = c1 * expansion

        if self.has_expansion:
            self.expansion_conv = stem(c1, expanded_c, kernel_size=k, stride=s)
            self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        else:
            self.project_conv = stem(c1, c2, kernel_size=k, stride=s)

        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expansion_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class MBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = stem(c1, expanded_c, kernel_size=1, stride=1)
        self.dw_conv = stem(expanded_c, expanded_c, kernel_size=k, stride=s, groups=expanded_c)
        self.se = SqueezeExcite_efficientv2(expanded_c, expanded_c, se_ration) if se_ration > 0 else nn.Identity()
        self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        # 先用1x1的卷积增加升维
        result = self.expansion_conv(x)
        # 再用一般的卷积特征提取
        result = self.dw_conv(result)
        # 添加se模块
        result = self.se(result)
        # 再用1x1的卷积降维
        result = self.project_conv(result)
        # 如果使用shortcut连接，则加入dropout操作
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            # shortcut就是到残差结构，输入输入的channel大小相等，这样就能相加了
            result += x

        return result
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/f8df63d4e21d5115d42067d4a6f2f999.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

加入 stem，FusedMBConv，MBConv 这三个模块

![](https://i-blog.csdnimg.cn/blog_migrate/6231b018bf8a5e5dfea6ed666b417e67.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为 yolov5s\_EfficientNetv2.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/13a417bb098e0390c8b5617081a4914b.png)

然后根据EfficientNetv2的网络结构来修改配置文件。

![](https://i-blog.csdnimg.cn/blog_migrate/7b5b893f126ab3f8273cc3a5c56d5d59.png)

接下来我们详细讲解一下每个Stage

（1）Stage0：Conv3×3

![](https://i-blog.csdnimg.cn/blog_migrate/00497b12b74a1af0085f566347bc33c1.png)

这一行的yaml参数应该如下： \[-1，1，stem，\[24，3，2\] \]

 *  24：out\_channel
 *  3： kernel\_size
 *  2：stride

（2）Stage1：Fused-MBConv1，k3×3

![](https://i-blog.csdnimg.cn/blog_migrate/d78b16190fb0e79ff7e8f844ad961533.png)

这一行的yaml参数应该如下： \[-1, 2, FusedMBConv, \[24, 3, 1, 1, 0\]\]

 *  Fused-MBConv1的1：expansion=1（不升维）
 *  24： out\_channel
 *  3： kernel\_size
 *  1：stride
 *  1：expansion
 *  0：SE\_ration

（3）Stage2：Fused-MBConv4，k3×3

![](https://i-blog.csdnimg.cn/blog_migrate/4ec890ccfe3f5221ebdca5733ffbafbc.png)

Stage2有4个模块：第一个模块是stride=2，后面三个模块是三次重复的stride都是1，所以yaml应该写为：  
第一个的stride为2：  
\[-1, 1, FusedMBConv, \[48, 3, 2, 4, 0\]\]  
后面三个的stride为1：  
\[-1, 3, FusedMBConv, \[48, 3, 1, 4, 0\]\]

 *  Fused-MBConv4的4：expansion=4（升维4倍）
 *  48： out\_channel
 *  3： kernel\_size
 *  2 / 1：stride
 *  4：expansion
 *  0：SE\_ration

（4）Stage3：Fused-MBConv4，k3×3

这块和上面同理，只是输出通道有变化。yaml应该写为：  
第一个的stride为2：  
\[-1, 1, FusedMBConv, \[64, 3, 2, 4, 0\]\]  
后面三个的stride为1：  
\[-1, 3, FusedMBConv, \[64, 3, 1, 4, 0\]\]

（5）Stage 4：MBConv4， k3x3，SE0.25 

![](https://i-blog.csdnimg.cn/blog_migrate/d12e16da994cb5cc15b875374f227d12.png)

Stage4有6个模块：第一个模块是stride=2，后面五个模块是五次重复的stride都是1，所以yaml应该写为：  
第一个的stride为2：  
\[-1, 1, MBConv, \[128, 3, 2, 4, 0.25\]\]  
后面五个的stride为1：  
\[-1, 5, MBConv, \[128, 3, 1, 4, 0.25\]\]

 *  MBConv4的4：expansion=4（升维4倍）
 *  128： out\_channel
 *  3： kernel\_size
 *  2 / 1：stride
 *  4：expansion
 *  0.25：SE\_ration

（6）Stage5：MBConv6，k3×3， SE0.25

![](https://i-blog.csdnimg.cn/blog_migrate/d9ceed4174dcdeba089c3c6f9489c417.png)

Stage5有9个模块：第一个模块是stride=2，后面八个模块是八次重复的stride都是1，所以yaml应该写为：

第一个的stride为2：  
\[-1, 1, MBConv, \[160, 3, 2, 6, 0.25\]\]  
后面八个的stride为1：  
\[-1, 8, MBConv, \[160, 3, 1, 6, 0.25\]\]

 *  MBConv6的6：expansion=4（升维4倍）
 *  160： out\_channel
 *  3： kernel\_size
 *  2 / 1：stride
 *  6：expansion
 *  0.25：SE\_ration

（7）Stage6：MBConv6，k3×3，SE0.25

这块和上面同理，只是输出通道和模块数有变化。yaml应该写为：

第一个的stride为2：  
\[-1, 1, MBConv, \[256, 3, 2, 4, 0.25\]\]  
后面十四个的stride为1：  
\[-1, 14, MBConv, \[256, 3, 1, 4, 0.25\]\]

注意，我们不需要stage7，因为我们只需要进行特征提取，不需要进行分类  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1dd36e2ec843d9644a06ea35a8fc2876.png)

yolov5s\_EfficientNetv2.yaml文件修改后完整代码如下：

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
  [[-1, 1, stem, [24, 3, 2]], #0 p1/2

   [-1, 2, FusedMBConv, [24, 3, 1, 1, 0]], # 1- last is use SE

   [-1, 1, FusedMBConv, [48, 3, 2, 4, 0]], # 2-p2/4
   [-1, 3, FusedMBConv, [48, 3, 1, 4, 0]], # 3

   [-1, 1, FusedMBConv, [64, 3, 2, 4, 0]], # 4-p3/8
   [-1, 3, FusedMBConv, [64, 3, 1, 4, 0]], # 5

   [-1, 1, MBConv, [128, 3, 2, 4, 0.25]], # 6-p4/16  last is use SE and ratio
   [-1, 5, MBConv, [128, 3, 1, 4, 0.25]], # 7

   [-1, 1, MBConv, [160, 3, 2, 6, 0.25]], # 8
   [-1, 8, MBConv, [160, 3, 1, 6, 0.25]], # 9

   [-1, 1, MBConv, [272, 3, 2, 4, 0.25]], # 10-p5/64
   [-1, 14, MBConv, [272, 3, 1, 4, 0.25]], # 11

   [-1, 1, SPPF, [1024, 5]], #12
#   [-1, 1, SPP, [1024, [5, 9, 13]]],
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]], # 13
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 14
   [[-1, 9], 1, Concat, [1]],  # 15 cat backbone P4
   [-1, 3, C3, [512, False]],  # 16

   [-1, 1, Conv, [256, 1, 1]], # 17
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 18
   [[-1, 7], 1, Concat, [1]],  # 19 cat backbone P3
   [-1, 3, C3, [256, False]],  # 20 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]], # 21
   [[-1, 17], 1, Concat, [1]],  # 22 cat head P4
   [-1, 3, C3, [512, False]],  # 23 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]], # 24
   [[-1, 13], 1, Concat, [1]],  # 25 cat head P5
   [-1, 3, C3, [1024, False]],  # 26 (P5/32-large)

   [[20, 23, 26], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_EfficientNetv2.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/071ea9f79fc45bccee1d18830f86bf32.png)

![](https://i-blog.csdnimg.cn/blog_migrate/2ba2f45b79da0fe1185b1fea89320549.png)

然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/e98b41475e57e8a579bc15b5a82daaa7.png)

这样就成功啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'yolov5s\_EfficientNetv2.yaml '，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/7eaeba30d149ae9d824955e668206e83.png)

PS：今天训练完对比了一下，发现速度是提高了1个多小时，但精读下降4个点，感觉没想象中那么好，不是很值。

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

![](https://i-blog.csdnimg.cn/blog_migrate/1fc629b34aa8be590ddef433e722b9b5.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[YOLOv5_5_ MobileNetV3]: https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501
[YOLOv5_6_ ShuffleNetV2]: https://blog.csdn.net/weixin_43334693/article/details/131008642?spm=1001.2014.3001.5501
[YOLOv5_7_SimAM]: https://blog.csdn.net/weixin_43334693/article/details/131031541?spm=1001.2014.3001.5501
[YOLOv5_8_SOCA]: https://blog.csdn.net/weixin_43334693/article/details/131053284?spm=1001.2014.3001.5501
[_EfficientNet]: #%F0%9F%9A%80%20%E4%B8%80%E3%80%81ShuffleNet%E4%BB%8B%E7%BB%8D
[1.1 EfficientNet V1]: #1.1%C2%A0ShuffleNet%20V1
[1.2 EfficientNet V2]: #1.2%C2%A0EfficientNet%20V2
[_YOLOv5_EfficientNetv2]: #%F0%9F%9A%80%20%E4%BA%8C%E3%80%81YOLOv5%E7%BB%93%E5%90%88EfficientNetv2
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_EfficientNetv2]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg_]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0%C2%A0
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[6_EfficientNetV1_]: https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5502
[7_EfficientNetV2_]: https://blog.csdn.net/weixin_43334693/article/details/131195722?spm=1001.2014.3001.5501
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