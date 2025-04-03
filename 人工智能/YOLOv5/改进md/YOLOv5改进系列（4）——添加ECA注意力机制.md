#### ![](https://i-blog.csdnimg.cn/blog_migrate/1555f335ec9129d453e82ade8df021b1.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/c74cbf21723c662b64c914e2628cc63d.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

![](https://i-blog.csdnimg.cn/blog_migrate/827e44ac92dc6a0b8625623c9cc993e4.gif)

目录

[🚀一、ECA注意力机制原理 ][ECA_]

[1.1 ECA方法介绍 ][1.1 ECA_]

[1.2 SE和ECA网络结构的对比][1.2 SE_ECA]

[1.3 ECA实现过程][1.3 ECA]

[🚀二、添加ECA注意力机制方法（单独加） ][ECA_ 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加ECA模块][common.py_ECA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀三、添加C3\_CA注意力机制方法（在C3模块中添加）][C3_CA_C3]

[第①步：在common.py中添加ECABottleneck和C3\_ECA模块][common.py_ECABottleneck_C3_ECA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/8fba1cffeee9685b8e9f07476fcbd7b6.gif)

## 🚀一、ECA注意力机制原理 

> 论文题目：《ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks》
> 
> 原文地址：[ECA-Net][]
> 
> 代码实现：[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks 开源代码][ECA-Net_ Efficient Channel Attention for Deep Convolutional Neural Networks][GitHub - BangguWu/ECANet: Code for ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks][GitHub - BangguWu_ECANet_ Code for ECA-Net_ Efficient Channel Attention for Deep Convolutional Neural Networks][ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks 开源代码][ECA-Net_ Efficient Channel Attention for Deep Convolutional Neural Networks]

### 1.1 ECA方法介绍 

ECA是通道注意力机制的一种实现形式，是基于SE的扩展。

作者认为SE block的两个FC层之间的降维是不利于channel attention的权重学习的，并且捕获所有通道之间的依存关系是效率不高且是不必要的。权重学习的过程应该直接一一对应。

ECA 注意力机制模块直接在全局平均池化层之后使用1x1卷积层，去除了全连接层。该模块避免了维度缩减，并有效捕获了跨通道交互。并且ECA只涉及少数参数就能达到很好的效果。

ECA通过一维卷积 layers.Conv1D来完成跨通道间的信息交互，卷积核的大小通过一个函数来自适应变化，使得通道数较大的层可以更多地进行跨通道交互。

自适应函数为： ![](https://latex.csdn.net/eq?k%3D%5Cleft%20%7C%20%5Cfrac%7Blog_%7B2%7D%28c%29%7D%7B%5Cgamma%20%7D%20+%5Cfrac%7Bb%7D%7B%5Cgamma%20%7D%20%5Cright%20%7C)，其中 ![\gamma =2,\,\,b=1](https://latex.csdn.net/eq?%5Cgamma%20%3D2%2C%5C%2C%5C%2Cb%3D1)

### 1.2 SE和ECA网络结构的对比 

![](https://i-blog.csdnimg.cn/blog_migrate/39431de262294b5a38ea6782f36c69cc.png)

<table> 
 <tbody> 
  <tr> 
   <td><strong>SEblock网络结构</strong></td> 
   <td><strong>ECA模块网络结构</strong></td> 
  </tr> 
  <tr> 
   <td>（1）global avg pooling产生1 ∗ 1 ∗ C 大小的feature maps</td> 
   <td>（1）global avg pooling产生1 ∗ 1 ∗ C 大小的feature maps</td> 
  </tr> 
  <tr> 
   <td>（2）两个fc层（中间有维度缩减）来产生每个channel的weight</td> 
   <td>（2）计算得到自适应的kernel_size</td> 
  </tr> 
  <tr> 
   <td></td> 
   <td>（3）应用kernel_size于一维卷积中，得到每个channel的weight</td> 
  </tr> 
 </tbody> 
</table>

### 1.3 ECA实现过程 

（1）将输入特征图经过全局平均池化，特征图从 \[h,w,c\] 的矩阵变成 \[1,1,c\] 的向量

（2）根据特征图的通道数计算得到自适应的一维卷积核大小 kernel\_size

（3）将 kernel\_size 用于一维卷积中，得到对于特征图的每个通道的权重

（4）将归一化权重和原输入特征图逐通道相乘，生成加权后的特征图

## 🚀二、添加ECA注意力机制方法（单独加） 

### 2.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将ECA类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml --> 新建一个文件夹，如yolov5s\_ECA.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加ECA模块 

将下面的ECA代码复制粘贴到common.py文件的末尾

```java
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1,c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/00d7520d24a6cab2b17c5957ae0a4231.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

然后把刚才加入的类ECA添加到这个注册表里面

![](https://i-blog.csdnimg.cn/blog_migrate/d8c11f2bcab37066d1509ba4f660d83b.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml文件，粘贴并重命名为 yolov5s\_ECA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/bac7bfa8e81288927fd2acc68440377f.png)

接着修改 yolov5s\_ECA.yaml  ，将ECA模块加到我们想添加的位置。

注意力机制可以添加在backbone，Neck，Head等部分，常见的有两种：一是在主干的 SPPF 前添加一层；二是将Backbone中的C3全部替换。

在这里我是用第一种：将 \[-1，1，ECA，\[1024\]\]添加到 SPPF 的上一层，下一节使用第二种。即下图中所示位置：

![](https://i-blog.csdnimg.cn/blog_migrate/19b693085339f63186a24110b4f5d72a.png)

同样的下面的head也得修改，p4，p5以及最后detect的总层数都得+1

![](https://i-blog.csdnimg.cn/blog_migrate/78edc7ac33a500e4734c8d4c80ea8610.png)

这里我们要把后面两个Concat的from系数分别由\[ − 1 , 14 \] ，\[ − 1 , 10 \]改为\[ − 1 , 15 \]，\[ − 1 , 11 \]。然后将Detect原始的from系数\[ 17 , 20 , 23 \]要改为\[ 18 , 21 , 24 \] 。

![](https://i-blog.csdnimg.cn/blog_migrate/0b6c22fa91f4a4c4f0c38314179421c5.png)

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_ECA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/e1062d7acdd940f3bf3080cd7e3245ad.png)

![](https://i-blog.csdnimg.cn/blog_migrate/bc216bf45c8ee64ee0fb56c64239a38b.png) 然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/a707636bfba1b892d86106561493567a.png)

找到ECA这一层，就说明我们添加成功啦！

可以看到params参数这里只有3，说明参数量真的很少呀。

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为'models/yolov5s\_ECA.yaml'，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/9665686e27da7ca6a7d981c790d8b4f8.png)

## 🚀三、添加C3\_CA注意力机制方法（在C3模块中添加） 

上面是单独加注意力层，接下来的方法是在C3模块中加入注意力层。

刚才也提到了，这个策略是将CA注意力机制添加到Bottleneck，替换Backbone中的所有C3模块。

（因为步骤和上面相同，所以接下来只放重要步骤噢~）

#### 第①步：在common.py中添加ECABottleneck和C3\_ECA模块 

将下面的代码复制粘贴到common.py文件的末尾

```java
class ECABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16, k_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.eca=ECA(c1,c2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        # out=self.eca(x1)*x1
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x1 * y.expand_as(x1)

        return x + out if self.add else out


class C3_ECA(C3):
    # C3 module with ECABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ECABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

在yolo.py的`parse_model`函数中，加入ECABottleneck，C3\_ECA这两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/0dc93e07f2f31629c3c568ffaaf5c616.png)

#### 第③步：创建自定义的yaml文件 

按照上面的步骤创建yolov5s\_C3\_ECA.yaml文件

![](https://i-blog.csdnimg.cn/blog_migrate/97d7439ac5fa9858ee1de0dcbb88f634.png)

替换4个C3模块，如下图所示

![](https://i-blog.csdnimg.cn/blog_migrate/66058c2c785a3e4a7a6b03e75c04c791.png)

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
   [-1, 3, C3_ECA, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3_ECA, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 3, C3_ECA, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3_ECA, [1024]],
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

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_C3\_ECA.yaml，然后运行

![](https://i-blog.csdnimg.cn/blog_migrate/8454034a0eff6877fa430bd5015988d4.png)

这样就OK啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

接下来的训练就和上面一样，不再叙述啦~

完结~撒花✿✿ヽ(°▽°)ノ✿

PS：今天训练了一下，我的评价是，ECA不如昨天的CA，mAP降了0.3

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
> CSDN： [【深度学习】(1) CNN中的注意力机制（SE、ECA、CBAM）][1_ CNN_SE_ECA_CBAM]
> 
> [手把手带你YOLOv5 (v6.1)添加注意力机制(二)（在C3模块中加入注意力机制）\_yolov5添加注意力机制\_迪菲赫尔曼的博客-CSDN博客][YOLOv5 _v6.1_C3_yolov5_-CSDN][手把手带你YOLOv5/v7 添加注意力机制（并附上30多种顶会Attention原理图）2023/2/11更新\_yolov5添加注意力机制\_迪菲赫尔曼的博客-CSDN博客][YOLOv5_v7 _30_Attention_2023_2_11_yolov5_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/2fd0307c49abd91d69d291a66c990a2c.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[ECA_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81ECA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E5%8E%9F%E7%90%86%C2%A0
[1.1 ECA_]: #1.1%20ECA%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D%C2%A0
[1.2 SE_ECA]: #1.2%20SE%E5%92%8CECA%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E7%9A%84%E5%AF%B9%E6%AF%94
[1.3 ECA]: #1.3%20ECA%E5%AE%9E%E7%8E%B0%E8%BF%87%E7%A8%8B
[ECA_ 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%B7%BB%E5%8A%A0ECA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%8D%95%E7%8B%AC%E5%8A%A0%EF%BC%89%C2%A0
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_ECA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[C3_CA_C3]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E6%B7%BB%E5%8A%A0C3_CA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%9C%A8C3%E6%A8%A1%E5%9D%97%E4%B8%AD%E6%B7%BB%E5%8A%A0%EF%BC%89
[common.py_ECABottleneck_C3_ECA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0ECABottleneck%E5%92%8CC3_ECA%E6%A8%A1%E5%9D%97
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[ECA-Net]: https://arxiv.org/abs/1910.03151
[ECA-Net_ Efficient Channel Attention for Deep Convolutional Neural Networks]: https://gitcode.net/mirrors/BangguWu/ECANet?utm_source=csdn_github_accelerator
[GitHub - BangguWu_ECANet_ Code for ECA-Net_ Efficient Channel Attention for Deep Convolutional Neural Networks]: https://github.com/BangguWu/ECANet
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
[1_ CNN_SE_ECA_CBAM]: https://blog.csdn.net/dgvv4/article/details/125112972?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168350459816800225558052%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168350459816800225558052&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-5-125112972-null-null.142%5Ev86%5Ekoosearch_v1,239%5Ev2%5Einsert_chatgpt&utm_term=%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187
[YOLOv5 _v6.1_C3_yolov5_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/124695537?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22124695537%22%2C%22source%22%3A%22weixin_43694096%22%7D
[YOLOv5_v7 _30_Attention_2023_2_11_yolov5_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/124443059?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168343384216800225517372%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168343384216800225517372&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124443059-null-null.142%5Ev86%5Ekoosearch_v1,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E6%B7%BB%E5%8A%A0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187