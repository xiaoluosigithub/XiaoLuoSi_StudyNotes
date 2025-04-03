![](https://i-blog.csdnimg.cn/blog_migrate/8f2863209475d917a84221d4db62f697.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/b3942530281078b4779f49ab3fe36b53.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/6e7df20d56b4203546bb53ba6b10bb0e.gif)

目录

[🌻一、BiFPN介绍][BiFPN]

[1.1 简介][1.1]

[1.2 BiFPN][]

[（1）跨尺度连接][1]

[（2）加权特征融合 ][2_]

[1.3 EfficientDet][]

[（1）模型框架][1 1]

[（2）复合缩放][2]

[🌻 二、添加方式1：Add操作][_1_Add]

[第①步：在common.py中添加BiFPN模块][common.py_BiFPN]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py ][train.py]

[🌻 三、添加方式2：Concat操作][_2_Concat]

[第①步：在common.py中添加BiFPN模块][common.py_BiFPN]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[第⑤步：修改train.py ][train.py]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/7bab22c5b35cfcf8cd8133998473ec0d.gif)

## 🌻一、BiFPN介绍 

> 直通车：[EfficientDet（EfficientNet+BiFPN）论文超详细解读（翻译＋学习笔记+代码实现）][EfficientDet_EfficientNet_BiFPN]

### 1.1 简介 

EfficientDet 是继 2019 年推出 EfficientNet 模型之后，Google 人工智能研究小组Tan Mingxing等人为进一步提高目标检测效率，以 EfficientNet 模型和双向特征加权金字塔网络 BiFPN为基础，于2020 年创新推出的新一代目标检测模型，在COCO数据集上吊打其他方法。

EfficientDet = Backbone(EfficientNet) \+ Neck(BiFPN) \+ Head(class + box)

![](https://i-blog.csdnimg.cn/blog_migrate/f63152f8474522d6925f4b869afcee4c.png)

### 1.2 BiFPN 

#### （1）跨尺度连接 

 *  移除那些只有一条输入边的节点，这是因为如果一个节点只有一条输入边而没有特征融合，那么它对以融合不同特征为目标的特征网络的贡献就比较小。这可以简化双向网络。
 *  如果原始输入节点和输出节点处于同一水平，就在它们之间增加一条额外的边，以便在不增加太多成本的情况下融合更多的特征。
 *  与PANet只有一条自上而下和一条自下而上的路径不同，BiFPN将每条双向(自上而下&自下而上)路径视为一个特征网络层，并多次重复同一层，以实现更高级别的特征融合。

![](https://i-blog.csdnimg.cn/blog_migrate/31e37c61fed47813e2d9b31177891b30.png)

#### （2）加权特征融合  

 *  无限融合：![](https://i-blog.csdnimg.cn/blog_migrate/13de39c03d429dcd6faf12231438143e.png)
 *  基于Softmax的融合：![](https://i-blog.csdnimg.cn/blog_migrate/981d76c598eb2ed5f2a7854b7a2eb9ab.png)
 *  快速归一化融合： ![](https://i-blog.csdnimg.cn/blog_migrate/ea60fd2463fb0268c79888611ec4af5c.png)

### 1.3 EfficientDet 

#### （1）模型框架 

![](https://i-blog.csdnimg.cn/blog_migrate/469730cf5b64158f00d57aa6ccbc86c5.png)

 *  backbone：EfficientNets
 *  特征网络：BiFPN

#### （2）复合缩放 

EfficientDet使用的是EfficientNet-B0到B7作为预训练模型，所以EfficientDet的系数ϕ的选择范围也是0~7。

Backbone network—主干网络

骨干网络采用和EfficientNet B0~B6相同的缩放系数，从而可以使用它们在ImageNet上的预训练模型。

BiFPN network—BiFPN 网络

对于BiFPN的深度 Dbifpn 采用线性变换的方式因为深度需要向下取整。对于宽度 Wbifpn采用指数变换的方式，采用网格搜索确定1.35作为宽度的缩放因子。

完整的缩放公式如下：

![](https://i-blog.csdnimg.cn/blog_migrate/9a0a8cde1657aa2e3073fc3ee0beac29.png)

Box/class prediction network—Box/class预测网络

宽度固定为和BiFPN的宽度相等即 Wpred=Wbifpn ，深度按下式进行线性变换：

![](https://i-blog.csdnimg.cn/blog_migrate/880487adb679eb587d7d38791d112553.png)

Input image resolution—输入图像分辨率

因为BiFPN中用到了level 3-7的特征，因此输入大小需要能被 2^7=128 除尽，因此输入分辨率按下式进行线性变换：

![](https://i-blog.csdnimg.cn/blog_migrate/efa66566e2ff97fd7fc7e117a3be9785.png)

## 🌻 二、添加方式1：Add操作 

#### 第①步：在common.py中添加BiFPN模块 

在common.py后面加入如下代码：

```java
# BiFPN 
# 两个特征图add操作
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


# 三个特征图add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/bdcd0f923eb7215494b4af24b68dc1f5.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat: 语句，在其后面加上BiFPN\_Add相关语句：

```java
# 添加bifpn_add结构
elif m in [BiFPN_Add2, BiFPN_Add3]:
    c2 = max([ch[x] for x in f])
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/c11e945035c43a7eabb312c52df9bec4.png)

#### 第③步：创建自定义的yaml文件 

这里的yaml文件将所有的Concat换成了BiFPN\_Add。

BiFPN\_Add本质是add操作，不是concat操作，因此BiFPN\_Add的各个输入层要求大小完全一致（通道数、feature map大小等）

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

# YOLOv5 v6.1 BiFPN head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, BiFPN_Add2, [256, 256]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, BiFPN_Add2, [128, 128]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 

   [-1, 1, Conv, [512, 3, 2]],  
   [[-1, 13, 6], 1, BiFPN_Add3, [256, 256]],  #v5s通道数是默认参数的一半
   [-1, 3, C3, [512, False]],  # 20 

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, BiFPN_Add2, [256, 256]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的 yolov5s\_BiFPN.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/c7e54e49ff9ecaa1a6b2104f11cf8067.png)

![](https://i-blog.csdnimg.cn/blog_migrate/bbb3841556393a4d7af383196d0ef0ef.png)

然后运行yolo.py 

![](https://i-blog.csdnimg.cn/blog_migrate/7765bfb3c676538c14ba4b36f3a531a9.png)

我们可以看到，所有的Concat已被换成了BiFPN\_Add

#### 第⑤步：修改train.py 

首先找到train.py文件里面的\# Optimizer

然后将BiFPN\_Add2 和 BiFPN\_Add3 函数中定义的w参数，加入g1

```java
# BiFPN_Concat
 elif isinstance(v, BiFPN_Add2) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)
 elif isinstance(v, BiFPN_Add3) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/1f1a43bc2734c1fd1cc2e989fe04d859.png)

刚加入的时候会报错，莫慌~

![](https://i-blog.csdnimg.cn/blog_migrate/5aee84154695aefd4c12c4b998640308.png)

这是没有导入包引起的啦~

我们可以直接导入 ：

![](https://i-blog.csdnimg.cn/blog_migrate/c2a3325595709b368a620f391c5f38af.png)

也可以加一个导包语句：

```java
from models.common import BiFPN_Add3, BiFPN_Add2
```

然后就可以开始训练了

![](https://i-blog.csdnimg.cn/blog_migrate/45160c51b73b21e1d75a08fdfda6ec11.png)

> --- 注意！---
> 
> 因为我使用的还是6.1版本，是可以直接在train.py进行修改的，但是在看了一些后期改进的文章，发现在yolov5-v7.0版本中，这个部分作者加入了智能优化器（smart\_optimizer）。  
> ![](https://i-blog.csdnimg.cn/blog_migrate/c5b9b4414bb956a37db9703c6287b285.png)
> 
> 我们ctrl+鼠标左键点击这个函数，进入之后可以发现optimizer这个函数进行了重构，之前的一重for循环被改成两重for。
> 
> ![](https://i-blog.csdnimg.cn/blog_migrate/9fe5c5417b84219218f2c7f63a6e0e4a.png) 另外，原来的 g\[0\] g\[1\] g\[2\] 被替换为g = \[\] \[\] \[\]
> 
>  *  新版将这个地方关于weight的顺序翻转了一下，这样就导致一个问题，只要不是bias或者weight no decay，那么就全都归结于weight with decay上。
>  *  与之前需要elif 进行判断Bi\_FPN进行模型的添加相比，这里不在需要添加判断条件了，因为最后的else会把 剩余非bias 和非weight nodecay 部分全部加到weight with decay上。
>  *  也就是说，添加其他Neck时，不需要额外对optimizer进行添加elif判断，也就实现了一个所谓智能的优化。
> 
> 所以7.0版本无需对参数g的修改，直接略过即可，智能优化器会对多余的部分进行自动增加权重。
> 
> （以上解析来自：[yolov5-7.0关于添加Bi\_FPN的探讨\_吃瓜太狼的博客-CSDN博客][yolov5-7.0_Bi_FPN_-CSDN]）

## 🌻 三、添加方式2：Concat操作 

#### 第①步：在common.py中添加BiFPN模块 

在common.py后面加入如下代码：

```java
# 结合BiFPN 设置可学习参数 学习不同分支的权重
# 两个分支concat操作
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
 
 
# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/7d0574e2f99a5022eed499a56506ce9f.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

再来修改yolo.py，在parse\_model函数中找到 elif m is Concat: 语句，在其后面加上BiFPN\_Concat相关语句：

```java
# 添加bifpn_concat结构
elif m in [Concat, BiFPN_Concat2, BiFPN_Concat3]:
    c2 = sum(ch[x] for x in f)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/9e6df223c9457bc314b18ae3e825cf12.png)

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
 
# YOLOv5 v6.0 BiFPN head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, BiFPN_Concat2, [1]],  # cat backbone P4 <--- BiFPN change
   [-1, 3, C3, [512, False]],  # 13
 
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, BiFPN_Concat2, [1]],  # cat backbone P3 <--- BiFPN change
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
 
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14, 6], 1, BiFPN_Concat3, [1]],  # cat P4 <--- BiFPN change
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
 
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, BiFPN_Concat2, [1]],  # cat head P5 <--- BiFPN change
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
 
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

和上面方法一样，我们直接运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/e837619417dd5be729332f6e7c43d968.png)

可以看到已经替换成功！

#### 第⑤步：修改train.py 

最后向优化器中添加BiFPN的权重参数，也和上面步骤一样。

```java
# BiFPN_Concat
        elif isinstance(v, BiFPN_Concat2) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)
        elif isinstance(v, BiFPN_Concat3) and hasattr(v, 'w') and isinstance(v.w, nn.Parameter):
            g1.append(v.w)
```

这样就OK啦~~~

> 本文参考：
> 
> [【YOLOv5-6.x】设置可学习权重结合BiFPN（Add操作）][YOLOv5-6.x_BiFPN_Add]
> 
> [【YOLOv5-6.x】设置可学习权重结合BiFPN（Concat操作）][YOLOv5-6.x_BiFPN_Concat]

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

![](https://i-blog.csdnimg.cn/blog_migrate/0aabe62540fedb2172684e721ade8d8f.gif)


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
[BiFPN]: #%F0%9F%8C%BB%E4%B8%80%E3%80%81BiFPN%E4%BB%8B%E7%BB%8D
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[1.2 BiFPN]: #1.2%C2%A0BiFPN
[1]: #%EF%BC%881%EF%BC%89%E8%B7%A8%E5%B0%BA%E5%BA%A6%E8%BF%9E%E6%8E%A5
[2_]: #%EF%BC%882%EF%BC%89%E5%8A%A0%E6%9D%83%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88%C2%A0
[1.3 EfficientDet]: #1.3%20EfficientDet
[1 1]: #%EF%BC%881%EF%BC%89%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6
[2]: #%EF%BC%882%EF%BC%89%E5%A4%8D%E5%90%88%E7%BC%A9%E6%94%BE
[_1_Add]: #%F0%9F%8C%BB%C2%A0%E4%BA%8C%E3%80%81%E6%B7%BB%E5%8A%A0%E6%96%B9%E5%BC%8F1%EF%BC%9AAdd%E6%93%8D%E4%BD%9C
[common.py_BiFPN]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0BiFPN%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%C2%A0%C2%A0
[_2_Concat]: #%F0%9F%8C%BB%C2%A0%E4%B8%89%E3%80%81%E6%B7%BB%E5%8A%A0%E6%96%B9%E5%BC%8F2%EF%BC%9AConcat%E6%93%8D%E4%BD%9C
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[EfficientDet_EfficientNet_BiFPN]: https://blog.csdn.net/weixin_43334693/article/details/131421396?spm=1001.2014.3001.5501
[yolov5-7.0_Bi_FPN_-CSDN]: https://blog.csdn.net/qq_41722524/article/details/129787478?ops_request_misc=&request_id=&biz_id=102&utm_term=bifpn%E5%BA%94%E7%94%A8%E5%88%B0yolov5&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-129787478.142%5Ev88%5Einsert_down1,239%5Ev2%5Einsert_chatgpt&spm=1018.2226.3001.4187
[YOLOv5-6.x_BiFPN_Add]: https://blog.csdn.net/weixin_43799388/article/details/124091648?spm=1001.2014.3001.5502
[YOLOv5-6.x_BiFPN_Concat]: https://blog.csdn.net/weixin_43799388/article/details/124220634?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-124220634-blog-116245197.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-124220634-blog-116245197.pc_relevant_antiscanv2&utm_relevant_index=2
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