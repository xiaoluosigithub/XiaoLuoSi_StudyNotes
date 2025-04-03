![](https://i-blog.csdnimg.cn/blog_migrate/424322369f9659ad4fe38789c23f7e7e.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/93a034d96e75828828b2576a3f19c40c.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/6e7df20d56b4203546bb53ba6b10bb0e.gif)

目录

[⭐一、小目标检测的介绍 ][Link 1]

[1.1 什么是小目标？][1.1]

[1.2 小目标检测遇到的问题][1.2]

[1.3 解决方法 ][1.3 _]

[1.4 YOLOv5中的优化方法][1.4 YOLOv5]

[⭐二、YOLOv5增加小目标层的方法 ][YOLOv5_]

[2.1 网络结构][2.1]

[2.2 添加步骤][2.2]

[第①步 创建自定义yaml文件][_yaml]

[第②步 验证是否添加成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/77513a336b035ba5d0f1c30f2fc337fc.gif)

## ⭐一、小目标检测的介绍 

### 1.1 什么是小目标？ 

（1）以物体检测领域的通用数据集COCO物体定义为例，小目标是指小于32×32个像素点（中物体是指32\*32-96\*96，大物体是指大于96\*96）。

（2）在实际应用场景中，通常更倾向于使用相对于原图的比例来定义：物体标注框的长宽乘积，除以整个图像的长宽乘积，再开根号，如果结果小于3%，就称之为小目标。

> 其他定义：
> 
>  *  目标边界框的宽高与图像的宽高比例小于一定值，通用值为0.1
>  *  边界框面积与图像面积之比的中位数在0.08%~0.58%之间
>  *  根据目标实际覆盖像素与图像总像素之间比例对小目标进行定义

### 1.2 小目标检测遇到的问题 

（1）大小目标混合的场合

在这种场合中，一张图片上有少数的大目标，有小目标。

常见的问题有：

 *  能够准确地检测到大目标，但检测不到小目标
 *  小目标的recall 率很低，大量的小目标检测不到，被遗漏
 *  小目标的数量太多，模型对小目标总是的支持不够

（2）只有小目标的场合

在这种场合中，一张图片上全是小目标。

常见的问题有：

 *  小目标的recall 率很低，大量的小目标检测不到，被遗漏
 *  小目标的数量太多，模型对小目标总数的支持不够

### 1.3 解决方法 

（1）图像的缩放（数据角度的方法）

很直觉的一种方法，效果也不错。问题在于如果对整张图像进行放大，即上采样，训练的成本会大大增加。

（2）使用深度较浅的网络

小物体更容易被接受场较小的探测器预测。较深的网络具有较大的接受域，容易丢失关于较粗层中较小对象的一些信息。

（3）利用小目标周围的上下文信息（数据角度的方法）

超分辨率，指针对小目标的图像增强等。最典型的是利用生成对抗性网络选择性地提高小目标的分辨率。

（4）图像金字塔

较早提出对训练图片上采样出多尺度的图像金字塔。通过上采样能够加强小目标的细粒度特征，在理论上能够优化小目标检测的定位和识别效果。但基于图像金字塔训练卷积神经网络模型对计算机算力和内存都有非常高的要求。计算机硬件发展至今也难有胜任。故该方法在实际应用中极少。

（5）逐层预测

该方法对于卷积神经网络的每层特征图输出进行一次预测，最后综合考量得出结果。同样，该方法也需要极高的硬件性能。

（6）特征金字塔

参考多尺度特征图的特征信息，同时兼顾了较强的语义特征和位置特征。该方法的优势在于，多尺度特征图是卷积神经网络中固有的过渡模块，堆叠多尺度特征图对于算法复杂度的增加微乎其微。

（7）RNN思想

参考了RNN算法中的门限机制、长短期记忆等，同时记录多层次的特征信息。

### 1.4 YOLOv5中的优化方法 

主要有以下几个方法：

（1）增加小目标检测层（本文）

（2）Transformer Prediction Heads (TPH)集成到YOLOv5（待研究）

（3）将CBAM集成到YOLOv5（[点这里！][YOLOv5_2_CBAM]）

（4）用Bi-FPN替换PAN-Net（[点这里！][YOLOv5_12_Neck_BiFPN]）

## ⭐二、YOLOv5增加小目标层的方法 

### 2.1 网络结构 

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0cac4b4ca58f26f52fb371484445db57.png#pic_center)

上图是YOLOv5-6.0的网络结构，由 输入端+Backbone+Neck +Head 组成（详解点这里：[【YOLO系列】YOLOv5超详细解读（网络详解）][YOLO_YOLOv5]）

这里我们可以看到原始的YOLOv5有3个检测头：分别是 20x20 （大目标） 40x40（中目标） 80x80(小目标)，我们要增加小目标检测层，就可以在80 x 80 的上一步，也就是 160 x 160 尺寸增加。

![](https://i-blog.csdnimg.cn/blog_migrate/d094eb46a27573d792b90aa58d71d574.png)

增加了一层检测层之后，网络结构就变成了酱紫：

![](https://i-blog.csdnimg.cn/blog_migrate/55c0f7bab945474613611ad9e4f0f122.png)

红框：这部分删除。原始网络结构中Neck部分是没有160 x 160的特征图，那么我们可以对 80 x 80的特征图再进行一次上采样，这样就得到160 x 160的特征图。

原来这一部分是网络的最后端，检测头可以直接获取特征图，现在增加一层检测层后就不是最后一层了，需要从它的下侧获取特征图。

蓝框：这一部分是我们新增的检测层，随着箭头往下走，先将Neck中80 x 80的特征图经过上采样变成160 x 160，这样就可以和Backbone中第2层160 x 160特征图进行concat融合，得到一个160 x 160的特征图。再往右走，然后经过下采样，又获得80 x 80的特征图（ FPN + TPN）并经过C3传给检测头。

### 2.2 添加步骤 

#### 第①步 创建自定义yaml文件 

首先创建yolov5s\_add\_one\_layer.yaml文件。将下面一段代码粘贴上去：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80
depth_multiple: 0.67
width_multiple: 0.75

anchors:
  - [4,5, 8,10, 22,18]   #  P2/4
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
   [[-1, 1, Conv, [64, 6, 2, 2]],  #0 卷积层 [64，320，320 ]
   [-1, 1, Conv, [128, 3, 2]], #1   卷积层 [128，160，160]
   [-1, 3, C3, [128]],         #2   C3 [128，160，160]

   [-1, 1, Conv, [256, 3, 2]], #3   卷积层 [256，80，80]
   [-1, 6, C3, [256]],         #4   C3 [256，80，80]

   [-1, 1, Conv, [512, 3, 2]],  #5  卷积层 [512，40，40]
   [-1, 9, C3, [512]],          #6 C3 [512，40，40]

   [-1, 1, Conv, [1024, 3, 2]], #7  卷积层 [1024，20，20]
   [-1, 3, C3, [1024]],         #8  C3 [1024，20，20]
   [-1, 1, SPPF, [1024, 5]],    #9 SPPF  [1024，20，20]
  ]

head:
#neck
  #[512，20，20]
  [[-1, 1, Conv, [512, 1, 1]],  #10  卷积层  [512，20，20]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #11  上采样 [512，40，40]
   [[-1, 6], 1, Concat, [1]],      #12 Concat [1024，40，40]

   [-1, 3, C3, [512, False]],      #13 C3  [512，40，40]
   [-1, 1, Conv, [256, 1, 1]],     #14  卷积层  [256，40，40]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #15  上采样 [256，80，80]
   [[-1, 4], 1, Concat, [1]], #16 Concat  [512，80，80]
   #[-1, 3, C3, [256, False]],  # 被删了

   #下面是我们自己加的
   [-1, 3, C3, [256, False]], #17  C3 [256，80，80]
   [-1, 1, Conv, [128, 1, 1]], #18  卷积层  [128，80，80]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],   #19 上采样 [128，160，160]
   [[-1, 2], 1, Concat, [1]],  #20 Concat [512，160，160]

   #head
   [-1, 3, C3, [128, False]],  #21  C3 [128，160，160]
   [-1, 1, Conv, [128, 3, 2]], #22  卷积层  [128，80，80]
   [[-1, 18], 1, Concat, [1]], #23  Concat [512，160，160]
   [-1, 3, C3, [256, False]],  #24   C3  [256，160，160]
   [-1, 1, Conv, [256, 3, 2]],  #25  卷积层 [256，40，40]
   [[-1, 14], 1, Concat, [1]],  #26 Concat [512，160，160]

   [-1, 3, C3, [512, False]],  #27 C3  [512，40，40]
   [-1, 1, Conv, [512, 3, 2]], #28  卷积层 [512，20，20]
   [[-1, 10], 1, Concat, [1]],  #29  特征融合 [1024，20，20]
   [-1, 3, C3, [1024, False]],  #30 C3  [1027，20，20]

   [[21, 24, 27, 30], 1, Detect, [nc, anchors]],  # 将21, 24, 27, 30传入检测头
  ]
```

我们来分析一下这段代码，这里主要是改了两个部分。

（1）修改Anchor：增加一组较小的anchor

```java
#---原始的anchors--#
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

若输入图像尺寸=640X640，

 *  \# P3/8 对应的检测特征图大小为 80X80，用于检测大小在 8X8 以上的目标。
 *  \# P4/16对应的检测特征图大小为 40X40，用于检测大小在 16X16 以上的目标。
 *  \# P5/32对应的检测特征图大小为 20X20，用于检测大小在 32X32 以上的目标。

```java
#---修改后的anchors---#
anchors:
  - [4,5, 8,10, 22,18]   #  P2/4
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

 *  新增加的\# P2/4 对应的检测特征图大小为 160X160，用于检测大小在 4X4 以上的目标

（2）修改Head部分：增加了一层网络结构

```java
#---原始的head部分---#
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

YOLOv5中的Head包括Neck和Detect两部分。前两个阶段是向上concat，后两个阶段是向下concat，最后有三个检测层，分别是在17层下面、20层下面、23层下面。

```java
#---修改后的head部分---#
head:
#neck
  #[512，20，20]
  [[-1, 1, Conv, [512, 1, 1]],  #10  卷积层  [512，20，20]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #11  上采样 [512，40，40]
   [[-1, 6], 1, Concat, [1]],      #12 Concat [1024，40，40]

   [-1, 3, C3, [512, False]],      #13 C3  [512，40，40]
   [-1, 1, Conv, [256, 1, 1]],     #14  卷积层  [256，40，40]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], #15  上采样 [256，80，80]
   [[-1, 4], 1, Concat, [1]], #16 Concat  [512，80，80]
   #[-1, 3, C3, [256, False]],  # 被删了

   #下面是我们自己加的
   [-1, 3, C3, [256, False]], #17  C3 [256，80，80]
   [-1, 1, Conv, [128, 1, 1]], #18  卷积层  [128，80，80]
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],   #19 上采样 [256，160，160]
   [[-1, 2], 1, Concat, [1]],  #20 Concat [512，160，160]

   #head
   [-1, 3, C3, [128, False]],  #21  C3 [128，160，160]
   [-1, 1, Conv, [128, 3, 2]], #22  卷积层  [128，80，80]
   [[-1, 18], 1, Concat, [1]], #23  Concat [512，160，160]
   [-1, 3, C3, [256, False]],  #24   C3  [256，160，160]
   [-1, 1, Conv, [256, 3, 2]],  #25  卷积层 [256，40，40]
   [[-1, 14], 1, Concat, [1]],  #26 Concat [512，160，160]

   [-1, 3, C3, [512, False]],  #27 C3  [512，40，40]
   [-1, 1, Conv, [512, 3, 2]], #28  卷积层 [512，20，20]
   [[-1, 10], 1, Concat, [1]],  #29  特征融合 [1024，20，20]
   [-1, 3, C3, [1024, False]],  #30 C3  [1027，20，20]

   [[21, 24, 27, 30], 1, Detect, [nc, anchors]],  # 将21, 24, 27, 30传入检测头
  ]
```

![](https://i-blog.csdnimg.cn/blog_migrate/c9ac9111c330ab9ad18e70db9857be3c.png)

首先在第17层，先将Neck中80 x 80的特征图经过上采样变成160 x 160，使得特征图继续扩大。

然后在第20层，将获取到的大小为160X160的特征图和Backbone中第2层的160 x 160的特征图进行concat融合，得到一个160 x 160的特征图，以此获取更大的特征图进行小目标检测。

最后在第31层，即检测层，增加小目标检测层，一共使用四层\[21, 24, 27, 30\]进行检测。

#### 第②步 验证是否添加成功 

将yolo.py文件中的配置改为刚刚我们配置好的yolov5s\_add\_one\_layer.yaml，运行一下

![](https://i-blog.csdnimg.cn/blog_migrate/d49da9a63bbf0f0b2a87ab723927fb5e.png)

这样就添加成功了~

（我去，计算量那么大的吗？）

PS：关于小目标检测的实验等我换完合适的数据集再发对比结果~

> 本文参考：
> 
> [【Yolov5】Yolov5添加检测层，四层结构对小目标、密集场景更友好][Yolov5_Yolov5]
> 
> [ yoloV5 增加小目标检测层][yoloV5]

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

![](https://i-blog.csdnimg.cn/blog_migrate/8bcbf26d4201067d3431d3eb214d058e.gif)


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
[Link 1]: #%E2%AD%90%E4%B8%80%E3%80%81%E5%B0%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%9A%84%E4%BB%8B%E7%BB%8D%C2%A0
[1.1]: #1.1%20%E4%BB%80%E4%B9%88%E6%98%AF%E5%B0%8F%E7%9B%AE%E6%A0%87%EF%BC%9F
[1.2]: #1.2%20%E5%B0%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E9%81%87%E5%88%B0%E7%9A%84%E9%97%AE%E9%A2%98
[1.3 _]: #1.3%20%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95%C2%A0
[1.4 YOLOv5]: #1.4%C2%A0%C2%A0YOLOv5%E4%B8%AD%E7%9A%84%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95
[YOLOv5_]: #%E2%AD%90%E4%BA%8C%E3%80%81YOLOv5%E5%A2%9E%E5%8A%A0%E5%B0%8F%E7%9B%AE%E6%A0%87%E5%B1%82%E7%9A%84%E6%96%B9%E6%B3%95%C2%A0
[2.1]: #2.1%20%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[2.2]: #2.2%20%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[_yaml]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%20%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89yaml%E6%96%87%E4%BB%B6
[Link 2]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%20%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E6%B7%BB%E5%8A%A0%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[YOLO_YOLOv5]: https://blog.csdn.net/weixin_43334693/article/details/129312409?spm=1001.2014.3001.5502
[Yolov5_Yolov5]: https://blog.csdn.net/weixin_50006912/article/details/129122501?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168877977616782425157211%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168877977616782425157211&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-129122501-null-null.142%5Ev88%5Einsert_down1,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E5%B0%8F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%B1%82&spm=1018.2226.3001.4187
[yoloV5]: https://blog.csdn.net/m0_64298393/article/details/130932049?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1-130932049-blog-129122501.235%5Ev38%5Epc_relevant_anti_t3&spm=1001.2101.3001.4242.1&utm_relevant_index=4
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