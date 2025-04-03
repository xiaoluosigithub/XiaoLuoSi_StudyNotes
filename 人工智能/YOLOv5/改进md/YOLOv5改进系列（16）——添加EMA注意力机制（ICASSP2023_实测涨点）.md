![](https://i-blog.csdnimg.cn/blog_migrate/e2836096f998f3ccea99496ba074746c.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/50a5fc7231e6c8d25ae09488ce414c27.png) ![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

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

![](https://i-blog.csdnimg.cn/blog_migrate/6e7df20d56b4203546bb53ba6b10bb0e.gif)

目录

[ 🚀一、EMA介绍 ][_EMA_]

[1.1 简介][1.1]

[🚀二、在Neck端添加EMA注意力机制方法][Neck_EMA]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加EMA模块][common.py_EMA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[ 🚀三、在C3中添加EMA注意力机制方法][_C3_EMA]

[3.1 具体添加步骤][3.1]

[第①步：在common.py中添加C3\_EMA模块][common.py_C3_EMA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model 1]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/223b8b788845505bb310462ef23fa279.gif)

## 🚀一、EMA介绍 

>  *  论文题目：《Efficient Multi-Scale Attention Module with Cross-Spatial Learning》
>  *  原文链接：[\[2305.13563v1\] Efficient Multi-Scale Attention Module with Cross-Spatial Learning (arxiv.org)][2305.13563v1_ Efficient Multi-Scale Attention Module with Cross-Spatial Learning _arxiv.org]

### 1.1 简介 

通道或空间的显著有效性注意机制对产生更多可辨识的特征表示的显著效果，然而用通道降维对跨通道关系进行建模关系，可能会给提取深层视觉表征带来副作用。比如CA注意力机制，虽对精度有一定的提升但是由于需要对整个特征图进行注意力权重的计算，因此需要额外的计算导致消耗资源，另外无法捕捉通道之间长距离的依赖关系。  
深圳的航天科工2023年6月提出的新型的高效多尺度注意力EMA模块，该模块首先将部分通道维度重塑为批量维度，以避免通用卷积进行某种形式的降维；接着在每个并行子网络中构建局部的跨通道交互，利用一种新的跨空间学习方法融合两个并行子网络的输出特征图，设计了一个多尺度并行子网络来建立短和长依赖关系。EMA的网络结构图如图所示。

![](https://i-blog.csdnimg.cn/blog_migrate/6f8a54d5e4728326c5cfc0e0222265f9.png)

## 🚀二、在Neck端添加EMA注意力机制方法 

### 2.1 添加顺序 

（1）models/common.py \--> 加入新增的网络结构

（2） models/yolo.py  \--> 设定网络结构的传参细节，将EMA类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml  \--> 新建一个文件夹，如yolov5s\_EMA.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py  \--> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加EMA模块 

第一步，通过与CA类似的处理，将两个编码特征连接在图像高度方向上，并使其共享相同的1×1卷积，而不会降低1x1分支的维数。在将1x1卷积的输出分解为2个向量后，使用两个非线性Sigmid函数来拟合线性卷积上的2D二进制分布。为了在1×1分支中的两个平行路线之间实现不同的跨通道交互特征，通过简单的乘法将每组内的两个通道注意力图聚合在一起。

![](https://i-blog.csdnimg.cn/blog_migrate/095406f24294854f4e73d8ac48ae50a8.png)

第二步，3×3分支通过卷积捕获局部跨通道交互，以扩大特征空间。

![](https://i-blog.csdnimg.cn/blog_migrate/03a2884434ce8970450d3c84a5315074.png)

这样，EMA不仅对通道间信息进行编码以调整不同通道的重要性，而目将精确的空间结构信息保存到通道中。

将下面的EMA代码复制粘贴到common.py文件的末尾。

```java
#EMA
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor # 分组因子
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1) #softmax操作
        self.agp = nn.AdaptiveAvgPool2d((1, 1)) # 1×1平均池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # X平均池化层 h=1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Y平均池化层 w=1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups) # 分组操作
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0) # 1×1卷积分支 
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1) # 3×3卷积分支

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x) # 得到平均池化之后的h
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # 得到平均池化之后的w
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 先拼接，然后送入1×1卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x) # 3×3卷积分支
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/1dc2e622d45373c0a3318f846d37a1dd.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行。

![](https://i-blog.csdnimg.cn/blog_migrate/9d0d06eae68325dfdc175fd78bb2d8a8.png)

然后把刚才加入的类EMA添加到这个注册表里面：

![](https://i-blog.csdnimg.cn/blog_migrate/9e4442a1fed448c125167fae9b837c8f.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为 yolov5s\_EMA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/7e5d68c96c9e4872afedc29f4e2cd861.png)

接着修改 yolov5s\_EMA.yaml，将EMA模块加到我们想添加的位置。 这里先演示的是在Neck端添加。

复制下面代码，粘贴到刚才新创建的yaml文件。

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 1  # number of classes
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
   [-1, 1, EMA, [256]],  # 加入到小目标层后

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
   [-1, 1, EMA, [512]],  # 加入到中目标层后

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
   [-1, 1, EMA, [1024]],  # 加入到大目标层后

   [[18, 22, 26], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/c6a385e78a405bf180e3a28e431f339d.png)

这样就修改成功了~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的default改为yolov5s\_EMA.yaml，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/a476d059e7993f9e8a8e442ccbddd888.png)

## 🚀三、在C3中添加EMA注意力机制方法 

### 3.1 具体添加步骤 

#### 第①步：在common.py中添加C3\_EMA模块 

将下面的代码复制粘贴到common.py文件的末尾。

```java
#EMA
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor # 分组率
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1) # Softmax
        self.agp = nn.AdaptiveAvgPool2d((1, 1)) # 平均池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # x平均池化层 h=1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # y平均池化层 w=1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups) # 分组操作
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0) # 1×1卷积分支
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1) # 3×3卷积分支

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class C3_EMA3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.m1 = nn.ModuleList([EMA(2 * c_)])  # 添加在最后一个卷积之前

    def forward(self, x):
        return self.cv3(self.m1[0](torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)))


class C3_EMA2(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.m1 = nn.ModuleList([EMA(c1)])  # 添加在最后一个卷积之前

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(self.m1[0](x))), 1))


class C3_EMA1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.m1 = nn.ModuleList([EMA(c_)])  # 添加在最后一个卷积之前

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.m1[0](self.cv1(x))), self.cv2(x)), 1))
```

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

![](https://i-blog.csdnimg.cn/blog_migrate/14902645b839fdd05aa471d4f08a6bfd.png)

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

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3_EMA1, [128]],
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
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/6f8d549d98b7e000e0ca8b2cadd5d390.png)

> 代码参考：
> 
> [Yolov5/Yolov7改进---注意力机制：ICASSP2023 EMA基于跨空间学习的高效多尺度注意力、效果优于ECA、CBAM、CA | 小目标涨点明显\_AI小怪兽的博客-CSDN博客][Yolov5_Yolov7_---_ICASSP2023 EMA_ECA_CBAM_CA _ _AI_-CSDN]

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

这步同上，就不再过多叙述了~

PS：

（1）在我的数据集上，使用EMA比CA涨了1.3，还是很不错的。

CA：

![](https://i-blog.csdnimg.cn/blog_migrate/6a69b7276acab0773539045f6004e932.png)

EMA：

![](https://i-blog.csdnimg.cn/blog_migrate/6978387b1b3881e402fb85609a6401e8.png)

（2）在我的数据集上，两种方法精度差不多，但第一种比第二种方法精度略高一丝。

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

![](https://i-blog.csdnimg.cn/blog_migrate/c2945744543f6a9cfb94738dc004bfe9.gif)


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
[_EMA_]: #%C2%A0%F0%9F%9A%80%E4%B8%80%E3%80%81EMA%E4%BB%8B%E7%BB%8D%C2%A0%C2%A0
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[Neck_EMA]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%9C%A8backbone%E6%9C%AB%E7%AB%AF%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_EMA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[_C3_EMA]: #%C2%A0%F0%9F%9A%80%E4%B8%89%E3%80%81%E5%9C%A8C3%E4%B8%AD%E6%B7%BB%E5%8A%A0EMA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[3.1]: #3.1%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[common.py_C3_EMA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0C3_EMA%E6%A8%A1%E5%9D%97
[yolo.py_parse_model 1]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[2305.13563v1_ Efficient Multi-Scale Attention Module with Cross-Spatial Learning _arxiv.org]: https://arxiv.org/abs/2305.13563v1
[Yolov5_Yolov7_---_ICASSP2023 EMA_ECA_CBAM_CA _ _AI_-CSDN]: https://blog.csdn.net/m0_63774211/article/details/131371039?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169320953716800185877967%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=169320953716800185877967&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-20-131371039-null-null.142%5Ev93%5EchatsearchT3_2&utm_term=ema%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187
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