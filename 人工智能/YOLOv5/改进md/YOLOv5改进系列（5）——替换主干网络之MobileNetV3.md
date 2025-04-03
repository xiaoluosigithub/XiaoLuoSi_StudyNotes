![](https://i-blog.csdnimg.cn/blog_migrate/d6d6ce7162a9eef388dc70628b13bc97.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ec5c19dbbc67e8a7f976e96958850b8e.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

[YOLOv5改进系列（4）——添加ECA注意力机制][YOLOv5_4_ECA]

![](https://i-blog.csdnimg.cn/blog_migrate/458fed7578c2138ada4730afe675baa5.gif)

目录

[ 🚀 一、MobileNetV3原理 ][_ _MobileNetV3_]

[1.1 MobileNetV3简介][1.1 MobileNetV3]

[1.2 MobileNetV3相关技术][1.2 MobileNetV3]

[🚀 二、YOLOv5结合MobileNetV3\_small][_YOLOv5_MobileNetV3_small]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加MobileNetV3模块][common.py_MobileNetV3]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀 三、YOLOv5结合MobileNetV3\_large][_YOLOv5_MobileNetV3_large]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/9eeeb33bdf5ee45da0dba372ad5cae15.gif)

## 🚀 一、MobileNetV3原理 

> [【轻量化网络系列（3）】MobileNetV3论文超详细解读（翻译 ＋学习笔记+代码实现）][3_MobileNetV3_]

### 1.1 MobileNetV3简介 

MobileNetV3，是谷歌在2019年3月21日提出的轻量化网络架构，在前两个版本的基础上，加入神经网络架构搜索（NAS）和h-swish[激活函数][Link 2]，并引入SE通道注意力机制，性能和速度都表现优异，受到学术界和工业界的追捧。

引用大佬的描述：MobileNet V3 = MobileNet v2 + SE结构 + hard-swish activation +网络结构头尾微调

![](https://i-blog.csdnimg.cn/blog_migrate/fb76ea56c6be6cacf27d1882f05793be.png)

MobileNetV1&MobileNetV2&MobileNetV3总结

<table> 
 <tbody> 
  <tr> 
   <td><span><strong>MobileNetV1&nbsp;</strong></span></td> 
   <td><span><strong>MobileNetV2&nbsp;</strong></span></td> 
   <td><strong><span>MobileNetV3</span></strong></td> 
  </tr> 
  <tr> 
   <td> 
    <ul> 
     <li>标准卷积改为<span>深度可分离卷积</span>，降低计算量；</li> 
     <li>ReLU改为<span>ReLU6</span>；</li> 
     <li>引入<span>Width Multiplier(α)</span>和<span>Resolution Multiplier(ρ)</span>，调节模型的宽度（卷积核个数）和图像分辨率；</li> 
    </ul></td> 
   <td> 
    <ul> 
     <li><span><strong>采用线性瓶颈层：</strong></span>将深度可分离卷积中的1×1卷积后的ReLU替换成<span>线性激活函数</span>；</li> 
     <li><span><strong>采用反向残差结构：</strong></span> 
      <ul> 
       <li>引入<span>Expansion layer</span>，在进行深度分离卷积之前首先使用1×1卷积进行<span>升维</span>；</li> 
       <li>引入<span>Shortcut</span>结构，在升维的1×1卷积之前与深度可分离卷积中的1×1卷积之后进行<span>shortcut连接</span>；</li> 
      </ul></li> 
    </ul></td> 
   <td> 
    <ul> 
     <li>采用增加了<span>SE机制</span>的Bottleneck模块结构；</li> 
     <li>使用了<span>一种新的激活函数h-swish(x)</span>替代MobileNetV2中的ReLU6激活函数；</li> 
     <li>网络结构搜索中，结合两种技术：<span>资源受限的NAS（platform-aware NAS）</span>与<span>NetAdapt</span>；</li> 
     <li>修改了MobileNetV2<span>网络端部最后阶段</span>；</li> 
    </ul></td> 
  </tr> 
 </tbody> 
</table>

### 1.2 MobileNetV3相关技术 

（1）引入MobileNetV1的深度可分离卷积  
（2）引入MobileNetV2的具有线性瓶颈的倒残差结构  
（3）引入基于squeeze and excitation结构的轻量级注意力模型(SE)  
（4）使用了一种新的激活函数h-swish(x)  
（5）网络结构搜索中，结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt  
（6）修改了MobileNetV2网络端部最后阶段

更多介绍，还是看上面的链接吧~

## 🚀 二、YOLOv5结合MobileNetV3\_small 

### 2.1 添加顺序 

之前在讲添加注意力机制时我们就介绍过改进网络的顺序，替换主干网络也是大同小异的。  
（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将MobileNetV3类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  修改现有模型结构配置文件

 *  当引入新的层时，要修改后续的结构中的from参数
 *  当仅替换主千网络时，要注意特征图的变换，/8，/16，/32

（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加MobileNetV3模块 

将以下代码复制粘贴到common.py文件的末尾

```java
# Mobilenetv3Small
# ——————MobileNetV3——————

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # 学习到的每一channel的权重
        return x * y


class conv_bn_hswish(nn.Module):
    """
    This equals to
    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish()
        )
    """

    def __init__(self, c1, c2, stride):
        super(conv_bn_hswish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = h_swish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MobileNetV3(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        # 输入通道数=扩张通道数 则不进行通道扩张
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # 否则 先进行通道扩张
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/0f300622aef56b074440c8db4c90beb4.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

加入h\_sigmoid，h\_swish，SELayer，conv\_bn\_hswish，MobileNetV3五个模块

![](https://i-blog.csdnimg.cn/blog_migrate/206978b5c3d82c4049fc3297a41e2b4e.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为yolov5s\_MobileNetv3.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/fbfcfa37eca9f95202029749c35b49eb.png) 然后根据MobileNetv3的网络结构来修改配置文件。

![](https://i-blog.csdnimg.cn/blog_migrate/d287832eb55b060e76dc6ad2aebe64d6.png)

根据网络结构我们可以看出MobileNetV3模块包含六个参数\[out\_ch, hidden\_ch, kernel\_size, stride, use\_se, use\_hs\]：

 *  out\_ch：输出通道
 *  hidden\_ch：表示在Inverted residuals中的扩张通道数
 *  kernel\_size：卷积核大小
 *  stride：步长
 *  use\_se：表示是否使用 SELayer，使用了是1，不使用是0
 *  use\_hs：表示使用 h\_swish 还是 ReLU，使用h\_swish是1，使用 ReLU是0

修改的时候，需要注意/8，/16，/32等位置特征图的变换

![](https://i-blog.csdnimg.cn/blog_migrate/720b12b6470a18e8fb944e2e270ac16d.png)

同样的，head部分这几个concat的层也要做修改：

![](https://i-blog.csdnimg.cn/blog_migrate/216a3b07c747878fd863626ec9cfdf14.png)

yaml文件修改后代码如下：

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

   # Mobilenetv3-small backbone
   # MobileNetV3_InvertedResidual [out_ch, hid_ch, k_s, stride, SE, HardSwish]
backbone:
  # [from, number, module, args]
  [[-1, 1, conv_bn_hswish, [16, 2]],             # 0-p1/2   320*320
   [-1, 1, MobileNetV3, [16,  16, 3, 2, 1, 0]],  # 1-p2/4   160*160
   [-1, 1, MobileNetV3, [24,  72, 3, 2, 0, 0]],  # 2-p3/8   80*80
   [-1, 1, MobileNetV3, [24,  88, 3, 1, 0, 0]],  # 3        80*80
   [-1, 1, MobileNetV3, [40,  96, 5, 2, 1, 1]],  # 4-p4/16  40*40
   [-1, 1, MobileNetV3, [40, 240, 5, 1, 1, 1]],  # 5        40*40
   [-1, 1, MobileNetV3, [40, 240, 5, 1, 1, 1]],  # 6        40*40
   [-1, 1, MobileNetV3, [48, 120, 5, 1, 1, 1]],  # 7        40*40
   [-1, 1, MobileNetV3, [48, 144, 5, 1, 1, 1]],  # 8        40*40
   [-1, 1, MobileNetV3, [96, 288, 5, 2, 1, 1]],  # 9-p5/32  20*20
   [-1, 1, MobileNetV3, [96, 576, 5, 1, 1, 1]],  # 10       20*20
   [-1, 1, MobileNetV3, [96, 576, 5, 1, 1, 1]],  # 11       20*20
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [96, 1, 1]],  # 12                         20*20
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 13         40*40
   [[-1, 8], 1, Concat, [1]],  # cat backbone P4            40*40
   [-1, 3, C3, [144, False]],  # 15                         40*40

   [-1, 1, Conv, [144, 1, 1]], # 16                         40*40
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],# 17          80*80
   [[-1, 3], 1, Concat, [1]],  # cat backbone P3            80*80
   [-1, 3, C3, [168, False]],  # 19 (P3/8-small)            80*80

   [-1, 1, Conv, [168, 3, 2]], # 20                         40*40
   [[-1, 16], 1, Concat, [1]], # cat head P4                40*40
   [-1, 3, C3, [312, False]],  # 22 (P4/16-medium)          40*40

   [-1, 1, Conv, [312, 3, 2]], # 23                         20*20
   [[-1, 12], 1, Concat, [1]], # cat head P5                20*20
   [-1, 3, C3, [408, False]],  # 25 (P5/32-large)           20*20

   [[19, 22, 25], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_MobileNetv3.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/fe488b0f36e0713eb979e45aefbe619e.png)

![](https://i-blog.csdnimg.cn/blog_migrate/008ef10603867108e57f4b718e0832ad.png)然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/ac6dd7c6b7ca1b97f9573d837caa6d42.png)

我们和原始的yolov5s.py进行对比

![](https://i-blog.csdnimg.cn/blog_migrate/62e5e6fa080c19981c5d9cfed80681cf.png)

可以看到替换主干网络为MobileNetV3之后层数变多了，可以学习到更多的特征；参数量由原来的700多万减少为500多万，大幅度减少了；GFLOPs由16.6变为12.2。

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'models/yolov5s\_MobileNetv3.yaml'，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/fb344167d3827b7d3e32fa54611f8667.png)

## 🚀 三、YOLOv5结合MobileNetV3\_large 

MobileNetV3\_large和MobileNetV3\_small区别在于yaml文件中head中concat连接不同，深度因子和宽度因子不同。

接下来我们就直接改动yaml的部分，其余参考上面步骤。

#### 第③步：创建自定义的yaml文件 

同样，首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为yolov5s\_MobileNetv3\_large.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/b1698bfe13b2e5e5281fd275dc029c28.png)

然后根据MobileNetv3的网络结构来修改配置文件。

![](https://i-blog.csdnimg.cn/blog_migrate/87a269c16392f66acfd6d4564e40e28f.png)

修改后代码如下：

```java
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

  [[-1, 1, conv_bn_hswish, [16, 2]],                  # 0-p1/2
   [-1, 1, MobileNetV3, [ 16,  16, 3, 1, 0, 0]],  # 1-p1/2
   [-1, 1, MobileNetV3, [ 24,  64, 3, 2, 0, 0]],  # 2-p2/4
   [-1, 1, MobileNetV3, [ 24,  72, 3, 1, 0, 0]],  # 3-p2/4
   [-1, 1, MobileNetV3, [ 40,  72, 5, 2, 1, 0]],  # 4-p3/8
   [-1, 1, MobileNetV3, [ 40, 120, 5, 1, 1, 0]],  # 5-p3/8
   [-1, 1, MobileNetV3, [ 40, 120, 5, 1, 1, 0]],  # 6-p3/8
   [-1, 1, MobileNetV3, [ 80, 240, 3, 2, 0, 1]],  # 7-p4/16
   [-1, 1, MobileNetV3, [ 80, 200, 3, 1, 0, 1]],  # 8-p4/16
   [-1, 1, MobileNetV3, [ 80, 184, 3, 1, 0, 1]],  # 9-p4/16
   [-1, 1, MobileNetV3, [ 80, 184, 3, 1, 0, 1]],  # 10-p4/16
   [-1, 1, MobileNetV3, [112, 480, 3, 1, 1, 1]],  # 11-p4/16
   [-1, 1, MobileNetV3, [112, 672, 3, 1, 1, 1]],  # 12-p4/16
   [-1, 1, MobileNetV3, [160, 672, 5, 1, 1, 1]],  # 13-p4/16
   [-1, 1, MobileNetV3, [160, 960, 5, 2, 1, 1]],  # 14-p5/32   原672改为原算法960
   [-1, 1, MobileNetV3, [160, 960, 5, 1, 1, 1]],  # 15-p5/32
  ]
# YOLOv5 v6.0 head
head:
  [ [ -1, 1, Conv, [ 256, 1, 1 ] ],
    [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
    [ [ -1, 13], 1, Concat, [ 1 ] ],  # cat backbone P4
    [ -1, 1, C3, [ 256, False ] ],  # 13

    [ -1, 1, Conv, [ 128, 1, 1 ] ],
    [ -1, 1, nn.Upsample, [ None, 2, 'nearest' ] ],
    [ [ -1, 6 ], 1, Concat, [ 1 ] ],  # cat backbone P3
    [ -1, 1, C3, [ 128, False ] ],  # 17 (P3/8-small)

    [ -1, 1, Conv, [ 128, 3, 2 ] ],
    [ [ -1, 20 ], 1, Concat, [ 1 ] ],  # cat head P4
    [ -1, 1, C3, [ 256, False ] ],  # 20 (P4/16-medium)

    [ -1, 1, Conv, [ 256, 3, 2 ] ],
    [ [ -1, 16 ], 1, Concat, [ 1 ] ],  # cat head P5
    [ -1, 1, C3, [ 512, False ] ],  # 23 (P5/32-large)

    [ [ 23, 26, 29 ], 1, Detect, [ nc, anchors ] ],  # Detect(P3, P4, P5)
  ]
```

网络运行结果：

![](https://i-blog.csdnimg.cn/blog_migrate/2cccdc85bb7f2a84da56b7a4a47f9316.png)

我们可以看到MobileNetV3-large模型比MobileNetV3-small多了更多的MobileNet\_Block结构，残差倒置结构中通道数维度也增大了许多，速度比YOLOv5s慢将近一半，但是参数变少，效果介乎MobileNetV3-small和YOLOv5s之间，可以作为模型对比，凸显自己模型优势。

PS：如果训练之后发现掉点纯属正常现象，因为轻量化网络在提速减少计算量的同时会降低精度。

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

> 本文参考：
> 
> b站：[【【手把手带你实战YOLOv5-进阶篇】YOLOv5 替换主干网络——以MobileNet为例】][YOLOv5-_YOLOv5 _MobileNet]
> 
> CSDN：[YOLOv5/v7 更换骨干网络之 MobileNetV3\_迪菲赫尔曼的博客-CSDN博客][YOLOv5_v7 _ MobileNetV3_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/859d83f235819855d1d65c3ba9373e6e.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[_ _MobileNetV3_]: #%C2%A0%F0%9F%9A%80%20%E4%B8%80%E3%80%81MobileNetV3%E5%8E%9F%E7%90%86%C2%A0
[1.1 MobileNetV3]: #1.1%C2%A0MobileNetV3%E7%AE%80%E4%BB%8B
[1.2 MobileNetV3]: #1.2%C2%A0MobileNetV3%E7%9B%B8%E5%85%B3%E6%8A%80%E6%9C%AF
[_YOLOv5_MobileNetV3_small]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%B7%BB%E5%8A%A0ECA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%8D%95%E7%8B%AC%E5%8A%A0%EF%BC%89%C2%A0
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0%C2%A0
[common.py_MobileNetV3]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[_YOLOv5_MobileNetV3_large]: #%F0%9F%9A%80%20%E4%B8%89%E3%80%81YOLOv5%E7%BB%93%E5%90%88MobileNetV3_large
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[3_MobileNetV3_]: https://blog.csdn.net/weixin_43334693/article/details/130834849?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22130834849%22%2C%22source%22%3A%22weixin_43334693%22%7D
[Link 2]: https://so.csdn.net/so/search?q=%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0&spm=1001.2101.3001.7020
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
[YOLOv5-_YOLOv5 _MobileNet]: ?vd_source=725f2b2a52500df1eaed63206ebe0ab2
[YOLOv5_v7 _ MobileNetV3_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/128522041?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168482817016800217283178%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168482817016800217283178&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-5-128522041-null-null.142%5Ev87%5Einsert_down28v1,239%5Ev2%5Einsert_chatgpt&utm_term=mobilenetv3%E4%B8%8Eyolov5%E7%BB%93%E5%90%88&spm=1018.2226.3001.4187