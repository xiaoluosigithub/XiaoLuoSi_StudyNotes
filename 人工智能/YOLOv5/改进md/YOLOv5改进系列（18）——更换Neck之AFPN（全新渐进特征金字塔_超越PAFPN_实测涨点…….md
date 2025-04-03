![](https://i-blog.csdnimg.cn/blog_migrate/82eee1d6e31ca69af7b52b6d4fa63964.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/db5b217e3e95f152f6d6db2bbface2c9.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/3d0a0a91e7a91cef9cbf6b7245bd6108.gif)

目录

[🚀一、AFPN介绍 ][AFPN_]

[1.1 简介][1.1]

[1.2 提取多级特征 ][1.2 _]

[1.3 渐进架构][1.3]

[1.4 自适应空间融合][1.4]

[1.5 实验][1.5]

[🚀二、更换AFPN的方法 ][AFPN_ 1]

[第①步：在common.py中添加AFPN模块][common.py_AFPN]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/735634204d490c98716751c36f279c24.gif)

## 🚀一、AFPN介绍 

>  *  论文题目：《AFPN: Asymptotic Feature Pyramid Network for Object Detection》
>  *  原文地址：[https://arxiv.org/abs/2306.15988v1][https_arxiv.org_abs_2306.15988v1]
>  *  代码地址： [GitHub - gyyang23/AFPN][GitHub - gyyang23_AFPN]

### ![](https://i-blog.csdnimg.cn/blog_migrate/530153b3ea43432514ff63b054fd46c3.png)1.1 简介 

多尺度特征在目标检测任务中对具有尺度方差的目标进行编码时具有重要意义。多尺度特征提取的一种常见策略是采用经典的自上而下和自下而上的特征金字塔网络。然而，这些方法遭受特征信息的丢失或退化，削弱了非相邻 Level 的融合效果。

本文使用了一种渐进特征金字塔网络AFPN来解决上述问题。在自下而上的特征提取过程中，首先通过组合两个具有不同分辨率的低层特征来启动融合过程。接着逐渐将高层特征纳入融合过程，最终融合主干网络的顶级特征。这种融合方式可以避免非相邻层次之间存在较大的语义差距。

在此过程中，模型学习了空间滤波冲突信息来抑制不一致性的方法，针对有利用价值的信息保留后再加以组合，自动地学习权重参数，以渐进的方式将低层特征与高层特征的语义信息和详细信息直接相互融合，避免了多级传输中的信息丢失或退化，提高特征尺度的不变性，并且引入的计算开销很小，实现简单。

![](https://i-blog.csdnimg.cn/blog_migrate/10a2f59ecc38e1e8bef228d73be9b0a3.png)

### 1.2 提取多级特征 

AFPN框架首先从主干网络的每个特征层提取最后一层特征，产生一组不同尺度的特征，表示为\{C2, C3, C4, C5\}。

接着将低层特征C2和C3输入到特征金字塔网络中，进行一次特征融合，然后添加C4，和之前添加的C2、C3一起做特征融合，最后添加C5，和之前添加的层继续做特征融合。最后产生一组多尺度特征\{P2, P3, P4, P5\}。

对于在Faster R-CNN框架上进行的实验，作者将Stride为2的卷积应用于P5，然后再应用Stride为1的卷积来生成P6，这确保了统一的输出。最后一组多尺度特征是\{P2, P3, P4, P5, P6\}，对应的特征Stride为\{4, 8, 16, 32, 64\}个像素。

### 1.3 渐进架构 

![](https://i-blog.csdnimg.cn/blog_migrate/d424e43f745252a7453f86d578c45900.png)

在主干网络自下而上的特征提取过程中，AFPN最初融合了低层特征，然后融合了深层特征，最后融合了最高层的特征，即最抽象的特征。

为了避免非相邻层次特征之间的语义差距大于相邻层次特征间的语义差距，导致的非相邻层次特征的融合效果较差问题，AFPN的架构采取渐进模式，每一层包含前一层的特征信息，这将使不同 Level 特征的语义信息在渐进融合过程中更加接近。

另外，为了对齐维度并为特征融合做准备，模型使用1×1卷积层和双线性插值方法对特征进行上采样，并根据所需的下采样率使用不同的卷积核和步长来执行下采样。

在特征融合之后，作者使用4个残差单元继续学习特征，这些残差单元类似于ResNet，每个残差单元包括2个3×3卷积。由于YOLO中只使用了3个 Level 的特征，因此没有8次上采样和8次下采样。

### 1.4 自适应空间融合 

在多级特征融合过程中，作者利用ASFF为不同 Level 的特征分配不同的空间权重，与基于元素和或级联的多层次特征融合方法不同，其核心思想是自适应地学习各尺度特征，映射融合的空间权重，通过学习权重参数的方式将不同层的特征融合到一起，可使网络自动学习过滤掉其他层的无用信息，保留有效信息，从而高效地融合特征。

![](https://i-blog.csdnimg.cn/blog_migrate/092ffc972c40e6ec8a075776daa03e75.png) 适应空间特征融合ASFF(Adaptive Spatial Feature Fusion)结构图

如上图所示，作者融合了3个层次的特征。让![{x_{ij}}^{n->l}](https://latex.csdn.net/eq?%7Bx_%7Bij%7D%7D%5E%7Bn-%3El%7D)表示从 Level n到 Level ![l](https://latex.csdn.net/eq?l)的位置(i,j)处的特征向量。结果特征向量表示为![l](https://latex.csdn.net/eq?l)，通过多级特征的自适应空间融合获得，并由特征向量的线性组合![{x_{ij}}^{1->l}](https://latex.csdn.net/eq?%7Bx_%7Bij%7D%7D%5E%7B1-%3El%7D)，![{x_{ij}}^{2->l}](https://latex.csdn.net/eq?%7Bx_%7Bij%7D%7D%5E%7B2-%3El%7D)和![{x_{ij}}^{3->l}](https://latex.csdn.net/eq?%7Bx_%7Bij%7D%7D%5E%7B3-%3El%7D)如下：

![](https://i-blog.csdnimg.cn/blog_migrate/cd5d483749ef91a053927545edff24f5.png)

### 1.5 实验 

（1）表I与MS-COCO VAL2017上不同特征金字塔网络的比较。

![](https://i-blog.csdnimg.cn/blog_migrate/52010022e6e7b7e733901dc3ffd341ce.png)

（2）表II与MS-COCO测试开发中不同特征金字塔网络的比较。

![](https://i-blog.csdnimg.cn/blog_migrate/1af6758b2dbacaabb1e005505bcb7293.png)

（3）表III 与其他两级目标探测器的比较。

![](https://i-blog.csdnimg.cn/blog_migrate/58ca6ba8bff13bd4a8e79265d67f6fa9.png)

（4）表IV AFPN对YOLOV5的贡献。

![](https://i-blog.csdnimg.cn/blog_migrate/c588ff9bc2a9e1aaaf55ec7dbff5bc58.png)

（5）表V 消融实验。

![](https://i-blog.csdnimg.cn/blog_migrate/886195188988dd5410d252b2ef30b0e6.png)

## 🚀二、更换AFPN的方法 

#### 第①步：在common.py中添加AFPN模块 

```java
class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""
# ---1.渐进架构部分（融合前的准备）--- #
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        # self.cv1 = Conv(c1, c2, 1)
        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # or model='bilinear' non-deterministic
        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # 如果下采样率为2，就用Stride为2的2×2卷积来实现2次下采样
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # 如果下采样率为4，就用Stride为4的4×4卷积来实现4次下采样

    def forward(self, x):
        # return self.upsample(self.cv1(x))
        return self.cv1(x)

# ---2.自适应空间融合（ASFF）--- #
class ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8
       
#如果是第0层
        if level == 0:
# self.stride_level_1调整level-1出来的特征图，通道调整为和level-0出来的特征图一样大小
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
#如果是第1层
        if level == 1:
# self.stride_level_0通道调整为和level-1出来的特征图一样大小
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # stride=2 下采样为2倍


# 两个卷积为了学习权重
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
# 用于调整拼接后的两个权重的通道
        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

# 如果在第0层
# level-0出来的特征图保持不变
# 调整level-1的特征图，使得其channel、width、height与level-0一致
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
# 如果在第1层，同上
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

# 将N*C*H*W的level-0特征图卷积得到权重，权重level_0_weight_v:N*256*H*W
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

# 将各个权重矩阵按照通道拼接
# levels_weight_v：N*3C*H*W
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)

# 将拼接后的矩阵调整，每个通道对应着不同的level_0_resized，level_1_resized的权重
        levels_weight = self.weights_levels(levels_weight_v)

# 在通道维度，对权重做归一化，也就是对于二通道tmp：tmp[0][0]+tmp[1][0]=1
        levels_weight = F.softmax(levels_weight, dim=1)

# 将levels_weight各个通道分别乘level_0_resized level_1_resized 
# 点乘用到了广播机制
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)

# ASFF3的运算流程同上
class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/f14c6ca1c37f297cdd757ae92f1e1806.png)

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到elif m is Concat: 语句，在其后面加上下面代码：

```java
elif m is ASFF2:
            c1, c2 = [ch[f[0]], ch[f[1]]], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is ASFF3:
            c1, c2 = [ch[f[0]], ch[f[1]], ch[f[2]]], args[0]
            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
```

如下图所示： 

![](https://i-blog.csdnimg.cn/blog_migrate/5cd425cffeb8993159cbc862ee74c1cc.png)

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

# YOLOv5 v6.1 backbone
backbone:
  # [from, repeats, module, args]
  [[-1, 1, Conv, [64, 3, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 6, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],
  ]  # 9

# YOLOv5 v6.1 head
head:
  [[4, 1, Conv, [128, 1, 1]], # 10 downsample backbone P3
   [6, 1, Conv, [256, 1, 1]], # 11 downsample backbone P4

   [[10, 11], 1, ASFF2, [128, 0]], # 12
   [[10, 11], 1, ASFF2, [256, 1]], # 13

   [-2, 1, C3, [128, False]], # 14
   [-2, 1, C3, [256, False]], # 15

   [9, 1, Conv, [512, 1, 1]], # 16 downsample backbone P5

   [[14, 15, 16], 1, ASFF3, [128, 0]], # 17
   [[14, 15, 16], 1, ASFF3, [256, 1]], # 18
   [[14, 15, 16], 1, ASFF3, [512, 2]], # 19

   [17, 1, C3, [256, False]],  # 20 (P3/8-small)
   [18, 1, C3, [512, False]],  # 21 (P4/16-medium)
   [19, 1, C3, [1024, False]],  # 22 (P5/32-large)
  [[20, 21, 22], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

OK！完成！

> 代码来源：
> 
> [YOLOv5/v7 引入渐进特征金字塔网络 AFPN 结构 | 《2023年6月28日最新发表》\_迪菲赫尔曼的博客-CSDN博客][YOLOv5_v7 _ AFPN _ _ _2023_6_28_-CSDN]

PS：

（1）本来没有准备放这个改进方法，一个是这个channel我研究了挺久，一直报错，后来直接借鉴大佬的了┗( T﹏T )┛。

再一个更重要的是，在我的数据集掉点了！┗|｀O′|┛ 嗷~~w(Д)w。。。我的评价是不如BiFPN。同样，也有很多友友反应掉点问题。它更大的优点可能就是参数少叭~后来看到大家讨论，这个改进可能更适合小数据集（如果合适可以试试，我没有验证。）

（2）昨晚我一个突发奇想，我做了如下改进：

![](https://i-blog.csdnimg.cn/blog_migrate/ea44cd7bf47a4f66c7b6e96b71ec9420.png)

就是在这三个位置加了EMA注意力。

在我的数据集上比CA+BiFPN组合涨了0.2，并且参数量更小，很nice！

![](https://i-blog.csdnimg.cn/blog_migrate/5fc017d54c35ced702f39e86a658da4d.jpeg)

大家如果有更好的改进方法可以在评论区留言呀~

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

![](https://i-blog.csdnimg.cn/blog_migrate/6bbe1359529aa85d32810d4c62b41197.gif)


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
[AFPN_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81AFPN%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0
[1.1]: #%E2%80%8B%E7%BC%96%E8%BE%911.1%20%E7%AE%80%E4%BB%8B
[1.2 _]: #1.2%20%E6%8F%90%E5%8F%96%E5%A4%9A%E7%BA%A7%E7%89%B9%E5%BE%81%C2%A0
[1.3]: #1.3%20%E6%B8%90%E8%BF%9B%E6%9E%B6%E6%9E%84
[1.4]: #1.4%20%E8%87%AA%E9%80%82%E5%BA%94%E7%A9%BA%E9%97%B4%E8%9E%8D%E5%90%88
[1.5]: #1.4%20%E5%AE%9E%E9%AA%8C
[AFPN_ 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%9B%B4%E6%8D%A2AFPN%E7%9A%84%E6%96%B9%E6%B3%95%C2%A0
[common.py_AFPN]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0BiFPN%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2306.15988v1]: https://arxiv.org/abs/2306.15988v1
[GitHub - gyyang23_AFPN]: https://github.com/gyyang23/AFPN
[YOLOv5_v7 _ AFPN _ _ _2023_6_28_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/131671240?spm=1001.2014.3001.5501
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