![](https://i-blog.csdnimg.cn/blog_migrate/e610f59569784b60e2026a8ddb8a3273.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/a23cde1135ece9a42d0f2397c6d9f5af.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/899ae1eb2322fb6f3abc4a918c8285ac.gif)

目录

[🚀 一、GhostNet介绍][_GhostNet]

[1.1 简介][1.1]

[1.2 基本单元][1.2]

[1.3 网络结构][1.3]

[🚀 二、YOLOv5结合GhostNet][_YOLOv5_GhostNet]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤][2.2]

[第①步：在common.py中添加GhostNet模块][common.py_GhostNet]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数 ][train.py_ _--cfg_]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/2cde1e80ed2a0d41d87ee173bea514ea.gif)

## 🚀 一、GhostNet介绍 

### 1.1 简介 

GhostNet 相比于普通的卷积神经网络在生成特征图时使用了更少的参数。它提出的动机是为了改善神经网络中特征图存在着冗余的现象。神经网络中的特征图存在着一定程度上的冗余，这些冗余的特征图一定程度上来说，也增强了网络对特征理解的能力，对于一个成功的模型来说这些冗余的特征图也是必不可少的。相比于有些轻量化网络去除掉这些冗余的特征图，GhostNet 选择低成本的办法来保留它们。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6717e5518447fa702c91822d77bfec41.jpeg#pic_center)

### 1.2 基本单元 

虽然 Shufflenet 和 Mobilenet 为了减少参数量使用了 1\*1 的逐点卷积方式，但是 Ghonstnet 的作者认为 1\*1 卷积还是会产生一定的计算量，并且发现许多的卷积神经网络并没有考虑到经过多次卷积后会存在特征冗余的现象。为了解决上述两个问题，作者提出了 Ghost 基本单元。

Ghost 基本单元采用了一系列的线性变换来生成特征图而不是采用卷积的方式生成特征图，这样可以减少网络的计算量。

图 a 为传统的卷积生成特征图的方式，图 b 为 Ghost 模块产生特征图的方式。

![](https://i-blog.csdnimg.cn/blog_migrate/c4baf27f791b5a17a28063677df29f21.png)

如图 b 所示，假设输入特征图的 shape 为\[5,5,6\]，首先对输入特征图使用 1\*1 卷积下降通道数，shape 变为 \[5,5,3\]；再使用 3\*3 深度卷积对每个通道特征图提取特征，shape 为 \[5,5,3\]，可以看作是经过前一层的一系列线性变换得到的；最后将两次卷积的输出特征图在通道维度上堆叠，shape 变为 \[5,5,6\]

GhostNet 模块在计算复杂度低，参数量少的情况下生成了和标准卷积一样大小的特征图。

### 1.3 网络结构 

![](https://i-blog.csdnimg.cn/blog_migrate/4d3ca0eac17a6498a53c3d3a193863d9.jpeg#pic_center)

整个Ghostnet都是由Ghost Bottlenecks进行组成的。

当一张图片输入到Ghostnet当中时：

（1）首先进入一个16通道的普通1x1卷积块（卷积+标准化+激活函数）。

（2）之后就开始Ghost Bottlenecks的堆叠了，利用Ghost Bottlenecks，最终获得了一个7x7x160的特征层（当输入是224x224x3的时候）。

（3）然后利用一个1x1的卷积块进行通道数的调整，此时可以获得一个7x7x960的特征层。

（4）接着进行一次全局平均池化，然后再利用一个1x1的卷积块进行通道数的调整，获得一个1x1x1280的特征层。

（5）最后平铺后进行全连接就可以进行分类了。

## 🚀 二、YOLOv5结合GhostNet 

### 2.1 添加顺序 

之前在讲添加注意力机制时我们就介绍过改进网络的顺序，替换主干网络也是大同小异的。

（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将GhostNet类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）

（3） models/yolov5\*.yaml --> 修改现有模型结构配置文件

 *  当引入新的层时，要修改后续的结构中的from参数
 *  当仅替换主千网络时，要注意特征图的变换，/8，/16，/32

（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加GhostNet模块 

这次比较特殊，因为在最新版本的[YOLOv5][YOLOv5 1]\-6.1源码中，作者已经加入了Ghost模块，在models/common.py 文件下

（就在Focus类的下面）

```java
# Ghost
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))

class G_bneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, midc, k=5, s=1, use_se = False):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()
        assert s in [1, 2]
        c_ = midc
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),              # Expansion
                                  Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity(),  # dw
                                  # Squeeze-and-Excite
                                  SeBlock(c_) if use_se else nn.Sequential(),
                                  GhostConv(c_, c2, 1, 1, act=False))   # Squeeze pw-linear

        self.shortcut = nn.Identity() if (c1 == c2 and s == 1) else \
                                                nn.Sequential(Conv(c1, c1, 3, s=s, p=1, g=c1, act=False), \
                                                Conv(c1, c2, 1, 1, act=False)) # 避免stride=2时 通道数改变的情况

    def forward(self, x):
        # print(self.conv(x).shape)
        # print(self.shortcut(x).shape)
        return self.conv(x) + self.shortcut(x)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/6c20c3b550e3a680d0ae22fbb68c1fdb.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

加入 G\_bneck 这个模块

![](https://i-blog.csdnimg.cn/blog_migrate/cd5d1d6dd73e7b2046bb71b8bcb4b0c1.png)

#### 第③步：创建自定义的yaml文件 

同样的，在models/hub/文件夹下，给出了yolo5s-ghost.yaml文件，因此我们直接使用即可

![](https://i-blog.csdnimg.cn/blog_migrate/9b69cd7f2f5e32cb79fa47242aaebc1f.png)

（你以为这篇文章就要这么水过去了吗✧ (≖ ‿ ≖)✧。。。

当然不可能啦！:.ﾟヽ(｡◕‿◕｡)ﾉﾟ.:｡+ﾟ）

参考了大佬[YOLOv5/v7 更换骨干网络之 GhostNet\_迪菲赫尔曼的博客-CSDN博客][YOLOv5_v7 _ GhostNet_-CSDN]的代码

接下来我们说一下yolo5l\_GhostNet.yaml 的写法

首先在models文件夹下复制yolov5l.yaml文件，粘贴并重命名为 yolo5l\_GhostNet.yaml 

![](https://i-blog.csdnimg.cn/blog_migrate/08ff1e7096ff85585e79b1ab386add76.png)

然后根据GhostNet的网络结构来修改配置文件。

![](https://i-blog.csdnimg.cn/blog_migrate/6f8532ef7e7a1f90b654f3e92b5d3ec7.png)

完整代码如下：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# Ghostnet backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [16, 3, 2, 1]],            # 0-P1/2  ch_out, kernel, stride, padding, groups 224*224*3
   [-1, 1, G_bneck, [16, 16, 3, 1]],        # 1  ch_out, ch_mid, dw-kernel, stride 112*112*16

   [-1, 1, G_bneck, [24, 48, 3, 2]],        # 2-P2/4   112*112*16
   [-1, 1, G_bneck, [24, 72, 3, 1]],        # 3         56*56*24

   [-1, 1, G_bneck, [40, 72, 3, 2, True]],  # 4-P3/8    56*56*24
   [-1, 1, G_bneck, [40, 120, 3, 1, True]], # 5         28*28*40

   [-1, 1, G_bneck, [80, 240, 3, 2]],        # 6-P4/16  28*28*40
   [-1, 3, G_bneck, [80, 184, 3, 1]],        # 7        14*14*80
   [-1, 1, G_bneck, [112, 480, 3, 1, True]], # 8        14*14*80
   [-1, 1, G_bneck, [112, 480, 3, 1, True]], # 9        14*14*80

   [-1, 1, G_bneck, [160, 672, 3, 2, True]], # 10-P5/32 14*14*112
   [-1, 1, G_bneck, [160, 960, 3, 1]],       # 11        7*7*160
   [-1, 1, G_bneck, [160, 960, 3, 1, True]], # 12        7*7*160
   [-1, 1, G_bneck, [160, 960, 3, 1]],       # 13        7*7*160
   [-1, 1, G_bneck, [160, 960, 3, 1, True]], # 14        7*7*160
   [-1, 1, Conv, [960]],                     # 15        7*7*160
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]], # 16
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 17
   [[-1, 9], 1, Concat, [1]],  # 18 cat backbone P4
   [-1, 3, C3, [512, False]],  # 19

   [-1, 1, Conv, [256, 1, 1]], # 20
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 21
   [[-1, 5], 1, Concat, [1]],  # 22 cat backbone P3
   [-1, 3, C3, [256, False]],  # 23 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]], # 24
   [[-1, 20], 1, Concat, [1]], # 25 cat head P4
   [-1, 3, C3, [512, False]],  # 26 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],  # 27
   [[-1, 15], 1, Concat, [1]],  # 28 cat head P5
   [-1, 3, C3, [1024, False]],  # 29 (P5/32-large)

   [[23, 26, 29], 1, Detect, [nc, anchors]],  # 30 Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py文件里面配置改为我们刚才自定义的yolo5l\_GhostNet.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/4609d76cdc4f96bb9765a7dc7db0fdf8.png)

![](https://i-blog.csdnimg.cn/blog_migrate/0079aa001b62eaebed3ce9cf1aca2acd.png)

然后运行yolo.py 

![](https://i-blog.csdnimg.cn/blog_migrate/88cd6d0f7cdfeb1a70650b0820d8d469.png)

这样就成功啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'yolo5l\_GhostNet.yaml'，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/fca77fabdf2ba52dd0ad5d782ae85c72.png)

## 🌟本人YOLOv5系列导航 

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀[YOLOv5源码][YOLOv5 2]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀[YOLOv5入门实践][YOLOv5 2]系列：

[YOLOv5入门实践（1）——手把手带你环境配置搭建][YOLOv5_1 1]

[YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集][YOLOv5_2_labelimg]

[YOLOv5入门实践（3）——手把手教你划分自己的数据集][YOLOv5_3]

[YOLOv5入门实践（4）——手把手教你训练自己的数据集][YOLOv5_4]

[YOLOv5入门实践（5）——从零开始，手把手教你训练自己的目标检测模型（包含pyqt5界面）][YOLOv5_5_pyqt5]

![](https://i-blog.csdnimg.cn/blog_migrate/f229096b7114e16b5f71c9450363ca0b.gif)


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
[_GhostNet]: #%F0%9F%9A%80%20%E4%B8%80%E3%80%81GhostNet%E4%BB%8B%E7%BB%8D
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[1.2]: #1.2%20%E5%9F%BA%E6%9C%AC%E5%8D%95%E5%85%83
[1.3]: #1.3%20%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[_YOLOv5_GhostNet]: #%F0%9F%9A%80%20%E4%BA%8C%E3%80%81YOLOv5%E7%BB%93%E5%90%88GhostNet
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_GhostNet]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg_]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0%C2%A0
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[YOLOv5 1]: https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020
[YOLOv5_v7 _ GhostNet_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/128523623?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168680616216800222883699%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168680616216800222883699&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-128523623-null-null.142%5Ev88%5Einsert_down1,239%5Ev2%5Einsert_chatgpt&utm_term=ghostnet%E7%94%A8%E5%9C%A8yolov5&spm=1018.2226.3001.4187
[YOLOv5 2]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
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