![](https://i-blog.csdnimg.cn/blog_migrate/0acfe4fcec0ed766a89d62928c947bdb.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ca1ead2abad763ad3868885064fb5dc6.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

[YOLOv5改进系列（4）——添加ECA注意力机制][YOLOv5_4_ECA]

[YOLOv5改进系列（5）——替换主干网络之 MobileNetV3][YOLOv5_5_ MobileNetV3]

![](https://i-blog.csdnimg.cn/blog_migrate/458fed7578c2138ada4730afe675baa5.gif)

目录

[🚀 一、ShuffleNet介绍][_ShuffleNet]

[1.1 ShuffleNet V1][]

[1.2 ShuffleNet V2][]

[🚀 二、YOLOv5结合ShuﬄeNet V2][_YOLOv5_Shu_eNet V2]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤][2.2]

[第①步：在common.py中添加ShuﬄeNet V2模块][common.py_Shu_eNet V2]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数 ][train.py_ _--cfg_]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/368d427f597a65d113fe49cff8f99adf.gif)

## 🚀 一、ShuffleNet介绍 

ShuffleNet系列轻量级卷积神经网络由旷世提出，也是非常有趣的轻量级卷积神经网络，它提出了通道混合的概念，改善了分组卷积存在的问题，加强各组卷积之间的特征交互和信息交流，在改善模型的特征提取方式的同时，增强特征提取的全面性。

### 1.1 ShuffleNet V1 

> [【轻量化网络系列（4）】ShuffleNetV1论文超详细解读（翻译 ＋学习笔记+代码实现）][4_ShuffleNetV1_]

简介

ShuffleNet V1是计算效率极高的CNN架构，该架构是专为计算能力非常有限（例如10-150 MFLOP）的移动设备设计的。新架构利用了两个新的操作，逐点组卷积和通道混洗，可以在保持准确性的同时大大降低计算成本。 ImageNet分类和MS COCO对象检测的实验证明了ShuffleNet V1优于其他结构的性能，例如在40个MFLOP的计算预算下，比最近的MobileNet \[12\]在ImageNet分类任务上的top-1错误要低（绝对7.8％）。在基于ARM的移动设备上，ShuffleNet V1的实际速度是AlexNet的13倍，同时保持了相当的准确性。

创新点 

 *  分组逐点卷积（pointwise group convolution）
 *  通道重排（channel shuffle）

网络模型结构

![](https://i-blog.csdnimg.cn/blog_migrate/6a410cf5b69f790932cdef27bf852cb6.png)

图（a)为一个Resdual block

 *  ①1×1卷积（降维）＋3×3深度卷积+1×1卷积（升维）
 *  ②之间有BN和ReLU
 *  ③最后通过add相加

图（b)为输入输出特征图大小不变的ShuffleNet Unit

 *  ①将第一个用于降低通道数的1×1卷积改为1×1分组卷积 + Channel Shuffle
 *  ②去掉原3×3深度卷积后的ReLU
 *  ③ 将第二个用于扩增通道数的1×1卷积改为1×1分组卷积

图（c)为输出特征图大小为输入特征图大小一半的ShuffleNet Unit

 *  ①将第一个用于降低通道数的1×1卷积改为1×1分组卷积 +Channel Shuffle
 *  ②令原3×3深度卷积的步长stride=2， 并且去掉深度卷积后的ReLU
 *  ③将第二个用于扩增通道数的1×1卷积改为1×1分组卷积
 *  ④shortcut上添加一个3×3平均池化层(stride=2)用于匹配特征图大小
 *  ⑤对于块的输出，将原来的add方式改为concat方式

### 1.2 ShuffleNet V2 

> [【轻量化网络系列（5）】ShuffleNetV2论文超详细解读（翻译 ＋学习笔记+代码实现）][5_ShuffleNetV2_]

简介

模型执行效率的准则不能完全取决于FLOPs，经常发现FLOPs差不多的两个模型的运算速度却不一样，因为FLOPs仅仅反映了模型的乘加次数，这种评价往往是片面的。影响模型运行速度的另一个指标也很重要，那就是MAC（memory access cost）内存访问成本。作者充分考虑了不同结构的MAC，从而设计了更加高效的网络模型ShuffleNet V2。

创新点

提出了四条实用准则：

 *  （1）使用“平衡卷积"（相等的通道数）
 *  （2）注意使用组卷积的成本
 *  （3）降低碎片化程度
 *  （4）减少逐元素操作

网络模型结构

![](https://i-blog.csdnimg.cn/blog_migrate/20eeda5ead93f96f1f42d8ca7d929f96.png)

 (c) ShuﬄeNet V2 的基本单元

 *  ①增加了Channel Split操作，实际上就是把输入通道分为两个部分。
 *  ②根据G1： 左边分支做恒等映射，右边的分支包含3个连续的卷积，并且输入和输出通道相同，每个分支中的卷积层的输入输出通道数都一致。
 *  ③根据G2： 两个1x1卷积不再是组卷积。
 *  ④根据G3： 减少基本单元数。因此有一个分支不做任何操作，直接做恒等映射。
 *  ⑤根据G4： 两个分支的输出不再是Add元素，而是concat在一起，紧接着是对两个分支concat结果进行channle shuffle，以保证两个分支信息交流。

(d) 用于空间下采样 (2×) 的 ShuffleNet V2 单元  
对于下采样模块，不再有channel split，每个分支都有stride=2的下采样，最后concat在一起后，特征图空间大小减半，但是通道数翻倍。

## 🚀 二、YOLOv5结合ShuﬄeNet V2  

### 2.1 添加顺序 

之前在讲添加注意力机制时我们就介绍过改进网络的顺序，替换主干网络也是大同小异的。  
（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将ShuﬄeNet V2类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  修改现有模型结构配置文件

 *  当引入新的层时，要修改后续的结构中的from参数
 *  当仅替换主千网络时，要注意特征图的变换，/8，/16，/32

（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加ShuﬄeNet V2模块 

将以下代码复制粘贴到common.py文件的末尾

```java
# 通道重排，跨group信息交流
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CBRM(nn.Module):           #conv BN ReLU Maxpool2d
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(CBRM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = ch_out // 2
        assert (self.stride != 1) or (ch_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(ch_in, ch_in, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(ch_in),

                nn.Conv2d(ch_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(ch_in if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 按照维度1进行split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/30567e2eec16c1f2128abfca629e78cd.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

加入 CBRM，Shuffle\_Block两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/1f0971ae25ffe0ed69ad8d6f824930d8.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为 yolov5s\_ShuffleNetV2.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/21db592cb03791d4b43d7bb194b98ef4.png)

然后根据ShuffleNetV2的网络结构来修改配置文件。

![](https://i-blog.csdnimg.cn/blog_migrate/9431c7c23510b14b1079440f88a6b489.png)

 yaml文件修改后代码如下：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 20  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  # Shuffle_Block: [out, stride]
  [[ -1, 1, CBRM, [ 32 ] ], # 0-P2/4
   [ -1, 1, Shuffle_Block, [ 128, 2 ] ],  # 1-P3/8
   [ -1, 3, Shuffle_Block, [ 128, 1 ] ],  # 2
   [ -1, 1, Shuffle_Block, [ 256, 2 ] ],  # 3-P4/16
   [ -1, 7, Shuffle_Block, [ 256, 1 ] ],  # 4
   [ -1, 1, Shuffle_Block, [ 512, 2 ] ],  # 5-P5/32
   [ -1, 3, Shuffle_Block, [ 512, 1 ] ],  # 6
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, C3, [256, False]],  # 10

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 2], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, C3, [128, False]],  # 14 (P3/8-small)

   [-1, 1, Conv, [128, 3, 2]],
   [[-1, 11], 1, Concat, [1]],  # cat head P4
   [-1, 1, C3, [256, False]],  # 17 (P4/16-medium)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 7], 1, Concat, [1]],  # cat head P5
   [-1, 1, C3, [512, False]],  # 20 (P5/32-large)

   [[14, 17, 20], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py文件里面配置改为我们刚才自定义的yolov5s\_ShuffleNetV2.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/2e816c07b6b3a0097adc858b0116ddf8.png)![](https://i-blog.csdnimg.cn/blog_migrate/44d83f592ef8390d145c243f186f056a.png)

然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/d74546a68e4d2b661bc48621895c8cab.png)

我们来和上次的MobileNet V3对比一下

![](https://i-blog.csdnimg.cn/blog_migrate/ac6dd7c6b7ca1b97f9573d837caa6d42.png)

可以看到替换主干网络为ShuffleNetV2之后层数变少了；参数量由原来的500多万减少为300多万，大幅度减少了；GFLOPs由12.2变为8.2。

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'models/yolov5s\_ShuffleNetV2.yaml '，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/65c3cce6dd5db6720d97ad01a49014fd.png)

PS： 我用的数据集有1442张，训练100轮，用时7个小时（更换前10个小时左右）。证明ShuffleNetV2的确能够大幅度提升速度，但是精度比原来掉了3个点，还是有点心疼的。

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

![](https://i-blog.csdnimg.cn/blog_migrate/a1c116c770f94ac258cbc31af5094bc3.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[YOLOv5_5_ MobileNetV3]: https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501
[_ShuffleNet]: #%F0%9F%9A%80%20%E4%B8%80%E3%80%81ShuffleNet%E4%BB%8B%E7%BB%8D
[1.1 ShuffleNet V1]: #1.1%C2%A0ShuffleNet%20V1
[1.2 ShuffleNet V2]: #1.2%C2%A0ShuffleNet%20V2
[_YOLOv5_Shu_eNet V2]: #%F0%9F%9A%80%20%E4%BA%8C%E3%80%81YOLOv5%E7%BB%93%E5%90%88Shu%EF%AC%84eNet%20V2
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_Shu_eNet V2]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg_]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0%C2%A0
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[4_ShuffleNetV1_]: https://blog.csdn.net/weixin_43334693/article/details/130905826?spm=1001.2014.3001.5501
[5_ShuffleNetV2_]: https://blog.csdn.net/weixin_43334693/article/details/130996003?spm=1001.2014.3001.5501
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