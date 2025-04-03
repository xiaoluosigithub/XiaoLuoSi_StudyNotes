#### ![](https://i-blog.csdnimg.cn/blog_migrate/40c0e409dfb0bf6390a91926f604a1e6.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/a7d7155fa763457942ca1cefb915e732.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

![](https://i-blog.csdnimg.cn/blog_migrate/47c9d5014467142c291cb451887a0b60.gif)

目录

[ 🚀一、CA注意力机制原理][_CA]

[1.1 SE和CBAM方法的不足][1.1 SE_CBAM]

[1.2 Coordinate Attention Block][]

[1.2.1 Coordinate Information Embedding：Coordinate信息嵌入][1.2.1 Coordinate Information Embedding_Coordinate]

[1.2.2 Coordinate Attention Generation ：Coordinate Attention生成][1.2.2 Coordinate Attention Generation _Coordinate Attention]

[1.3 CA的优势][1.3 CA]

[🚀二、添加CA注意力机制方法（单独加） ][CA_]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加CA模块][common.py_CA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀三、添加C3\_CA注意力机制方法（在C3模块中添加）][C3_CA_C3]

[第①步：在common.py中添加CABottleneck和C3\_CA模块][common.py_CABottleneck_C3_CA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/a9d376b75dfcfab704cbbf47071cdae7.gif)

## 🚀一、CA注意力机制原理 

> 论文题目：《Coordinate Attention for Efficient Mobile Network Design》
> 
> 论文地址：[https://arxiv.org/abs/2103.02907][https_arxiv.org_abs_2103.02907]
> 
> 代码实现：[ https://github.com/Andrew- Qibin/CoordAttention ][https_github.com_Andrew- Qibin_CoordAttention]

### 1.1 SE和CBAM方法的不足 

 *  SE（Squeeze-and-Excitation）：SE注意力机制在编码过程中只考虑编码通道间信息而忽略了位置信息的重要性
 *  CBAM（CBAM: Convolutional Block Attention Module）：使用卷积计算空间注意力来利用位置信息，但卷积只能捕获局部关系，而无法捕捉的长距离依赖关系。

### 1.2 Coordinate Attention Block 

![](https://i-blog.csdnimg.cn/blog_migrate/33bca2e7db978364c0de212e89b461fc.png#pic_center)

Coordinate Attention通过精确的位置信息对通道关系和长期依赖性进行编码，具体操作分为 Coordinate信息嵌入 和 Coordinate Attention生成 2个步骤。

#### 1.2.1 Coordinate Information Embedding：Coordinate信息嵌入 

将 CxHxW形状的输入特征图逐通道进行平均池化，使用（H，1）和（1，W）的池化核分别按X和Y轴方向进行池化，产生 CxHx1和 Cx1xW 形状的特征图。

![](https://i-blog.csdnimg.cn/blog_migrate/7c787df9d03432ce8bfe66f3aec13675.png)

通过这种方式所产生的一对方法感知特征图，可以使 CA 注意力能够在一个通道内捕获长距离的依赖关系，并且还有助于保留精确的位置信息，使网络能够更加准确的定位对象。

#### 1.2.2 Coordinate Attention Generation ：Coordinate Attention生成 

（1）将上述所提取到的特征图按空间维度进行拼接，拼接成 ![\frac{C}{r}](https://latex.csdn.net/eq?%5Cdpi%7B80%7D%20%5Cfrac%7BC%7D%7Br%7D)x1x(W+H) 形状的特征图，其中 r 用于控制块的减小率和SE中的作用相同。

（2）再将特征图经过F1卷积变换函数（1x1卷积）和非线性激活函数产生 f 中间特征图。然后将 f 按空间维度拆分成两个张量 fh 和 fw ，形状分别为![\frac{C}{r}](https://latex.csdn.net/eq?%5Cdpi%7B80%7D%20%5Cfrac%7BC%7D%7Br%7D)xH 和 ![\frac{C}{r}](https://latex.csdn.net/eq?%5Cdpi%7B80%7D%20%5Cfrac%7BC%7D%7Br%7D)xW，接着分别进行 Fh 和 Fw 卷积变换函数（1x1卷积）和 Sigmoid 激活函数，得到 gh 和 gw 坐标注意力。

（3）最后将gh 和 gw 与原输入进行相乘，得到与输入相同形状的输出。

![](https://i-blog.csdnimg.cn/blog_migrate/cf5f8a8a2223253166ac9868d91308b2.png#pic_center)

### 1.3 CA的优势 

 *  不仅考虑了通道信息，还考虑了方向相关的位置信息。
 *  足够的灵活和轻量，能够简单的插入到轻量级网络的核心模块中。
 *  可以作为预训练模型用于多种任务中，如检测和分割，均有不错的性能提升。

## 🚀二、添加CA注意力机制方法（单独加） 

### 2.1 添加顺序 

（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将CA（CoordAtt

）类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_CA.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加CA模块 

将下面的CA代码复制粘贴到common.py文件的末尾

```java
# CA
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

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x) # N*C*H*1
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # N*C*1*W
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/7293786c3d064ca1945d440e169a051a.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

然后把刚才加入的类CoordAtt添加到这个注册表里面

![](https://i-blog.csdnimg.cn/blog_migrate/526c491a0ed5208f27a4682263ac3f1a.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为yolov5s\_CA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/f98b6d8c80e22022bee67d06f95b6d7b.png)

接着修改yolov5s\_CA.yaml，将CA模块加到我们想添加的位置。

注意力机制可以添加在backbone，Neck，Head等部分， 常见的有两种：一是在主干的 SPPF 前添加一层；二是将Backbone中的C3全部替换。

在这里我是用第一种：将 \[-1，1，CA，\[1024\]\]添加到 SPPF 的上一层，下一节使用第二种。即下图中所示位置：

![](https://i-blog.csdnimg.cn/blog_migrate/fa2469b2161c53eb98717e574bb93af1.png)

同样的下面的head也得修改，p4，p5以及最后detect的总层数都得+1

![](https://i-blog.csdnimg.cn/blog_migrate/78edc7ac33a500e4734c8d4c80ea8610.png)

这里我们要把后面两个Concat的from系数分别由\[ − 1 , 14 \] ,\[ − 1 , 10 \]改为\[ − 1 , 15 \] ,\[ − 1 , 11 \]。然后将Detect原始的from系数\[ 17 , 20 , 23 \]要改为\[ 18 , 21 , 24 \] 。

![](https://i-blog.csdnimg.cn/blog_migrate/3fbb39efeb5b506a42408c36b2bffa33.png)

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_CA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/2ced3ec9609b6fe0c8ae8f7594ef81c9.png)

![](https://i-blog.csdnimg.cn/blog_migrate/414e954f669410a9047e68ad4c88db59.png)

然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/9b31b2dfdba4dd5bf5509c3f2b71a60f.png)

找到CA这一层，就说明我们添加成功啦！

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'models/yolov5s\_CA.yaml'，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/8ffae05c1f76c971ccfa80dd487ec47b.png)

## 🚀三、添加C3\_CA注意力机制方法（在C3模块中添加） 

上面是单独加注意力层，接下来的方法是在C3模块中加入注意力层。

刚才也提到了，这个策略是将CA注意力机制添加到Bottleneck，替换Backbone中的所有C3模块。

（因为步骤和上面相同，所以接下来只放重要步骤噢~）

#### 第①步：在common.py中添加CABottleneck和C3\_CA模块 

将下面的代码复制粘贴到common.py文件的末尾

```java
# CA
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


class CABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.ca=CoordAtt(c1,c2,ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // ratio)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x1)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x1 * a_w * a_h

        # out=self.ca(x1)*x1
        return x + out if self.add else out


class C3_CA(C3):
    # C3 module with CABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

在yolo.py的`parse_model`函数中，加入`CABottleneck，C3_CA这`两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/c5ff50462e9d303fe2e961c460029bd3.png)

#### 第③步：创建自定义的yaml文件 

按照上面的步骤创建yolov5s\_C3\_CA.yaml文件，替换4个C3模块

![](https://i-blog.csdnimg.cn/blog_migrate/f588381c874a5c7a741314d4551b0702.png)

代码如下：

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
   [-1, 3, C3_CA, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3_CA, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 3, C3_CA, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3_CA, [1024]],
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

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_C3\_CA.yaml，然后运行

![](https://i-blog.csdnimg.cn/blog_migrate/b43ef932d01b57e3e6a109e9734aacb4.png)

这样就OK啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

接下来的训练就和上面一样，不再叙述啦~

完结~撒花✿✿ヽ(°▽°)ノ✿

PS：今天训练的结果和比昨天CBAM的mAP增长了1.1，说明CA效果还是很不错的~

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

> 本文参考（感谢大佬们）：
> 
> b站：[【YOLOv5 v6.1添加SE,CA,CBAM,ECA注意力机制教学，即插即用】][YOLOv5 v6.1_SE_CA_CBAM_ECA]
> 
> CSDN： [CVPR 2021 | 即插即用！ CA：新注意力机制，助力分类/检测/分割涨点！\_Amusi（CVer）的博客-CSDN博客][CVPR 2021 _ _ CA_Amusi_CVer_-CSDN]
> 
> [手把手带你YOLOv5 (v6.1)添加注意力机制(二)（在C3模块中加入注意力机制）\_迪菲赫尔曼的博客-CSDN博客][YOLOv5 _v6.1_C3_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/f4fdd2e8b19aae21071c0c7bf033f764.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[_CA]: #%C2%A0%F0%9F%9A%80%E4%B8%80%E3%80%81CA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E5%8E%9F%E7%90%86
[1.1 SE_CBAM]: #1.1%20SE%E5%92%8CCBAM%E6%96%B9%E6%B3%95%E7%9A%84%E4%B8%8D%E8%B6%B3
[1.2 Coordinate Attention Block]: #1.2%20Coordinate%20Attention%20Block
[1.2.1 Coordinate Information Embedding_Coordinate]: #1.2.1%C2%A0Coordinate%20Information%20Embedding%EF%BC%9ACoordinate%E4%BF%A1%E6%81%AF%E5%B5%8C%E5%85%A5
[1.2.2 Coordinate Attention Generation _Coordinate Attention]: #1.2.2%20Coordinate%20Attention%20Generation%C2%A0%EF%BC%9ACoordinate%20Attention%E7%94%9F%E6%88%90
[1.3 CA]: #1.3%20CA%E7%9A%84%E4%BC%98%E5%8A%BF
[CA_]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%B7%BB%E5%8A%A0CA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%8D%95%E7%8B%AC%E5%8A%A0%EF%BC%89%C2%A0
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_CA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[C3_CA_C3]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E6%B7%BB%E5%8A%A0C3_CA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%9C%A8C3%E6%A8%A1%E5%9D%97%E4%B8%AD%E6%B7%BB%E5%8A%A0%EF%BC%89
[common.py_CABottleneck_C3_CA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0CABottleneck%E5%92%8CC3_CA%E6%A8%A1%E5%9D%97
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2103.02907]: https://arxiv.org/abs/2103.02907
[https_github.com_Andrew- Qibin_CoordAttention]: http://xn--https-rfa//github.com/Andrew-%20Qibin/CoordAttention
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
[YOLOv5 v6.1_SE_CA_CBAM_ECA]: https://blog.csdn.net/https://www.bilibili.com/video/BV1kS4y1c7Bm?vd_source=725f2b2a52500df1eaed63206ebe0ab2
[CVPR 2021 _ _ CA_Amusi_CVer_-CSDN]: https://blog.csdn.net/amusi1994/article/details/114559459?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168378218016800182145839%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168378218016800182145839&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-114559459-null-null.142%5Ev86%5Ekoosearch_v1,239%5Ev2%5Einsert_chatgpt&utm_term=ca%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187
[YOLOv5 _v6.1_C3_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/124695537?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22124695537%22%2C%22source%22%3A%22weixin_43694096%22%7D