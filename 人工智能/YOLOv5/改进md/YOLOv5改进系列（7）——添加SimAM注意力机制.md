![](https://i-blog.csdnimg.cn/blog_migrate/3e6cd2c9ce33d9ba0da93118d18fe04d.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/4e07a8af30c9d1cbf2c83b4b4c24004b.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

[YOLOv5改进系列（4）——添加ECA注意力机制][YOLOv5_4_ECA]

[YOLOv5改进系列（5）——替换主干网络之 MobileNetV3][YOLOv5_5_ MobileNetV3]

[YOLOv5改进系列（6）——替换主干网络之 ShuffleNetV2][YOLOv5_6_ ShuffleNetV2]

![](https://i-blog.csdnimg.cn/blog_migrate/3de901aaa0e7f412c1bd582bb2151a03.gif)

目录

[🚀一、SimAM介绍][SimAM]

[1.1 简介][1.1]

[1.2 方法][1.2]

[🚀二、在backbone末端添加SimAM注意力机制方法][backbone_SimAM]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加SimAM模块][common.py_SimAM]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀三、在C3后添加SimAM注意力机制方法][C3_SimAM]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/b315b902be65c3938cc8a94104f3c84e.gif)

## 🚀一、SimAM介绍 

>  *  论文题目：《SimAM: A Simple, Parameter-Free Attention [Module][] for Convolutional Neural Networks》
>  *  论文地址：[http://proceedings.mlr.press/v139/yang21o/yang21o.pdf][http_proceedings.mlr.press_v139_yang21o_yang21o.pdf]
>  *  代码地址：[GitHub - ZjjConan/SimAM: The official pytorch implemention of our ICML paper "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks".][GitHub - ZjjConan_SimAM_ The official pytorch implemention of our ICML paper _SimAM_ A Simple_ Parameter-Free Attention Module for Convolutional Neural Networks_.]

### 1.1 简介 

SimAM (Simple Attention Mechanism) 是中山大学在注意力机制方面的尝试，从神经科学理论出发，构建了一种能量函数挖掘神经元重要性，并对此推导出了解析解以加速计算。通过ImageNet分类、COCO检测与分割等任务验证了所提SimAM的灵活性与有效性。值得一提的是，所提SimAM是一种无参数注意力模块。

SimAM的设计思路源于SENet，但不同于SENet的复杂结构，SimAM只使用了一个全局池化层和几个全连接层。具体来说，我们在YOLOV5的倒数第二个卷积层后面加上一个全局池化层，将卷积层的输出向量变为标量。然后，我们将这个标量输入到两个全连接层中，以得到两个权重向量。这两个权重向量分别用于加权卷积层的特征图和辅助分类器的特征图。

### 1.2 方法 

之前的方法 

1.  （a）通道注意力：1D注意力，它对不同通道区别对待，对所有位置同等对待；
2.  （b）空间注意力：2D注意力，它对不同位置区别对待，对所有通道同等对待。

本文的方法

权值生成方法。现有注意力往往采用额外的子网络生成注意力权值，比如SE的GAP+FC+ReLU+FC+Sigmoid。更多注意力模块的操作、参数量可参考下表。总而言之，现有注意力的结构设计需要大量的工程性实验。我们认为：注意力机制的实现应当通过神经科学中的某些统一原则引导设计。

## 🚀二、在backbone末端添加SimAM注意力机制方法 

### 2.1 添加顺序 

（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将SimAM类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml --> 新建一个文件夹，如yolov5s\_SimAM.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加SimAM模块 

将下面的SimAM代码复制粘贴到common.py文件的末尾

```java
#SimAM
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/148f605e6adb0641447513c5bf5da164.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

然后把刚才加入的类SimAM添加到这个注册表里面：

![](https://i-blog.csdnimg.cn/blog_migrate/805d1bf7c09ca1d106a839a9a6203ae9.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml 文件，粘贴并重命名为 yolov5s\_SimAM.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/0caa9153595ca68ad12ad4d26e5b628f.png)

接着修改 yolov5s\_SimAM.yaml ，将SimAM模块加到我们想添加的位置。

这里我先介绍第一种，第一种是将SimAM模块放在backbone部分的最末端，这样可以使注意力机制看到整个backbone部分的特征图，将具有全局视野，类似于一个小transformer结构。

将 \[-1，1，SimAM，\[1024\]\]添加到 SPPF 的下一层。即下图中所示位置：

![](https://i-blog.csdnimg.cn/blog_migrate/706ecd341b443bc9f301c150e1abbe5a.png)

同样的下面的head也得修改：

![](https://i-blog.csdnimg.cn/blog_migrate/78edc7ac33a500e4734c8d4c80ea8610.png)

这里我们要把后面两个Concat的from系数分别由\[ − 1 , 14 \]，\[ − 1 , 10 \]改为\[ − 1 , 15 \]，\[ − 1 , 11 \]。

然后将Detect原始的from系数\[ 17 , 20 , 23 \]要改为\[ 18 , 21 , 24 \] 。

![](https://i-blog.csdnimg.cn/blog_migrate/0b6c22fa91f4a4c4f0c38314179421c5.png)

yolov5s\_SimAM.yaml完整代码：

```java
# YOLOv5 🚀 by YOLOAir, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

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
   [-1, 3, SimAM, [1024]], # 10
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 14

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 18 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 15], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 21 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 11], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 24 (P5/32-large)

   [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_SimAM.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/2a6c7f38ebad354059dc24c5694fc6a1.png)

![](https://i-blog.csdnimg.cn/blog_migrate/3d791acadafeeb23190d324b7c3032c6.png)

然后我们运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/7a179e637005811989d16cc17ab1028b.png)

哇哦~果真是无参数捏！

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到train.py文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为yolov5s\_SimAM.yaml，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/62864aab442d783d3bff0b55cb72e8de.png)

## 🚀三、在C3后添加SimAM注意力机制方法 

第二种是将SimAM放在backbone部分每个C3模块的后面，这样可以使注意力机制看到局部的特征，每层进行一次注意力，可以分担学习压力。

步骤和方法1相同，只是yaml文件不同。

所以接下来只放修改yaml文件的部分~

#### 第③步：创建自定义的yaml文件 

将SimAM模块放在每个C3模块的后面，要注意通道的变化。

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/2d7d22dd94f744bd75569b78f1fcbe82.png)

同样的，下面的head部分也要做相应的修改

（数的时候一定要认真，我今天上午弄错了，卡了半天55~）

![](https://i-blog.csdnimg.cn/blog_migrate/7c1876ebfd53e54fce5fb6d6cd689aeb.png)

第二种方法的 yolov5s\_SimAM.yaml完整代码：

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
   [-1, 3, SimAM, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 6, C3, [256]],
   [-1, 3, SimAM, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 9, C3, [512]],
   [-1, 3, SimAM, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 10-P5/32
   [-1, 3, C3, [1024]],
   [-1, 3, SimAM, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 13

  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 9], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 17

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 21 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 18], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 24 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 27 (P5/32-large)

   [[21, 24, 27], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

同样的方法，我们来运行一下yolo.py 

![](https://i-blog.csdnimg.cn/blog_migrate/36575254a8510ccb58b1b555c996a271.png)

OK~完成啦！

 PS：今天尝试加入SimAM注意力机制训练时间为4个多小时（比昨天轻量化网络还快？）；结果比之前尝试效果最佳的CA注意力机制还涨了近1个点，可以说非常奈斯了！

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

![](https://i-blog.csdnimg.cn/blog_migrate/f6ba02be26b6085f4e7ba1d4b96caa76.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[YOLOv5_5_ MobileNetV3]: https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501
[YOLOv5_6_ ShuffleNetV2]: https://blog.csdn.net/weixin_43334693/article/details/131008642?spm=1001.2014.3001.5501
[SimAM]: #%F0%9F%9A%80%E4%B8%80%E3%80%81SimAM%E4%BB%8B%E7%BB%8D
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[1.2]: #1.2%20%E6%96%B9%E6%B3%95
[backbone_SimAM]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%9C%A8backbone%E6%9C%AB%E7%AB%AF%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_SimAM]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[C3_SimAM]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E5%9C%A8C3%E5%90%8E%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[Module]: https://so.csdn.net/so/search?q=Module&spm=1001.2101.3001.7020
[http_proceedings.mlr.press_v139_yang21o_yang21o.pdf]: http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
[GitHub - ZjjConan_SimAM_ The official pytorch implemention of our ICML paper _SimAM_ A Simple_ Parameter-Free Attention Module for Convolutional Neural Networks_.]: https://github.com/ZjjConan/SimAM
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