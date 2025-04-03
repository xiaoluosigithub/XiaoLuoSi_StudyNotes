## ![](https://i-blog.csdnimg.cn/blog_migrate/48e585e38350110b9c34346851458120.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/2f71e907dd7066c3099f6ef050a4ba42.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/6e7df20d56b4203546bb53ba6b10bb0e.gif)

目录

[🌲一、激活函数的简介][Link 1]

[1.1 什么是激活函数？][1.1]

[1.2 激活函数有什么用？][1.2]

[🌲二、一些常见的激活函数][Link 2]

[🌻2.1 sigmoid函数][2.1 sigmoid]

[🌻2.2 tanh / 双曲正切激活函数][2.2 tanh _]

[🌻2.3 ReLU激活函数][2.3 ReLU]

[🌻2.4 SiLU ][2.4 SiLU]

[🌻2.5 swish激活函数][2.5 swish]

[🌻2.6 hardswish激活函数][2.6 hardswish]

[🌻2.7 Mish激活函数][2.7 Mish]

[🌻2.8 ELU激活函数][2.8 ELU]

[🌻2.9 AconC/MetaAconC激活函数][2.9 AconC_MetaAconC]

[🌻2.10 Softplus激活函数][2.10 Softplus]

[🌲三、更换激活函数的步骤][Link 3]

[第①步 打开activations.py文件，查看已有函数][_activations.py]

[第②步 在common.py中更换激活函数 ][_common.py_]

[🌲四、如何选择合适的激活函数][Link 4]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/c33f146aba4d74ba22a17dfd12431d32.gif)

## 🌲一、激活函数的简介 

### 1.1 什么是激活函数？ 

在接触到深度学习（Deep Learning）后，特别是神经网络中，我们会发现在每一层的神经网络输出后都会使用一个函数（比如sigmoid，tanh，Relu等等）对结果进行运算，这个函数就是激活函数（Activation Function）。那么为什么需要添加激活函数呢？如果不添加又会产生什么问题呢？

首先，我们知道神经网络模拟了人类神经元的工作机理，激活函数（Activation Function）是一种添加到人工神经网络中的函数，旨在帮助网络学习数据中的复杂模式。在神经元中，输入的input经过一系列加权求和后作用于另一个函数，这个函数就是这里的激活函数。激活函数是一种特殊的非线性函数，它能够在神经网络中使用，其作用是将输入信号转化成输出信号。类似于人类大脑中基于神经元的模型，它将神经元中的输入信号转换为一个有意义的输出，从而使得神经网络能够学习和识别复杂的模式。激活函数最终决定了是否传递信号以及要发射给下一个神经元的内容。在人工神经网络中，一个节点的激活函数定义了该节点在给定的输入或输入集合下的输出。

激活函数可以分为线性激活函数以及非线性激活函数，常用的激活函数有 Sigmoid、ReLU、Leaky ReLU 和 ELU 等。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4eff32a218577cc627cf4e464eff111b.png)

### 1.2 激活函数有什么用？ 

一句话总结：为了提高模型的表达能力。

激活函数能让中间输出多样化，从而能够处理更复杂的问题。如果不使用激活函数，那么每一层的输出都是上一层输入的线性函数，最后的输出也只是最开始输入数据的线性组合而已。而激活函数可以给神经元引入非线性因素，当加入到多层神经网络时，就可以让神经网络拟合任何线性函数或非线性函数，从而使得网络可以适合更多的非线性问题，而不仅仅是线性问题。

激活函数被定义为一个几乎处处可微的函数。

## 🌲二、一些常见的激活函数 

### 🌻2.1 sigmoid函数 

sigmoid 是使用范围最广的一类激活函数，具有指数函数形状，它在物理意义上最为接近生物神经元，是一个在生物学中常见的S型的函数，也称为S型生长曲线。此外，(0, 1) 的输出还可以被表示作概率，或用于输入的归一化，代表性的如Sigmoid交叉熵损失函数。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/0cd4ad0d906dc93c477e94ed8b32102f.png)

图像如下：![](https://i-blog.csdnimg.cn/blog_migrate/c05a80937c8285b73a3b0fd40f4d8d0d.png)

sigmoid优点：

（1） 值域在0和1之间

（2）函数具有非常好的对称性

（3）sigmoid的优点在于输出范围有限，所以数据在传递的过程中不容易发散。当然也有相应的缺点，就是饱和的时候梯度太小

（4）求导容易

sigmoid缺点：

（1）容易出现梯度消失

（2）函数输出并不是zero-centered（零均值）

（3）幂运算相对来讲比较耗时

### 🌻2.2 tanh / 双曲正切激活函数 

tanh激活函数又叫作双曲正切激活函数，在sigmoid基础上做出改进，与sigmoid相比，tanh输出均值为0，能够加快网络的收敛速度。然而，tanh同样存在梯度消失现象。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/3887e8b1a36b80091b77a6e9473fd9b8.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/e58d7204fef7f7410e92b4bf82e91938.png)

tanh优点：

（1）当输入较大或较小时，输出几乎是平滑的并且梯度较小，这不利于权重更新。tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好

（2）在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零。

tanh缺点：

（1）仍然存在梯度饱和的问题

（2）依然进行的是指数运算

### 🌻2.3 ReLU激活函数 

ReLU函数又称为修正线性单元（Rectified Linear Unit），是一种分段线性函数，通常指代以斜坡函数及其变种为代表的非线性函数。其弥补了sigmoid函数以及tanh函数的梯度消失问题，在目前的深度神经网络中被广泛使用。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/76f9a79ce2f903889b2cffee84d617d6.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/7a640919a29d8b463560d839eba4b2ce.png)

ReLU优点： 

（1）当输入为正时，导数为1，一定程度上改善了梯度消失问题，加速梯度下降的收敛速度

（2）计算速度快得多。ReLU 函数中只存在线性关系，因此它的计算速度比 sigmoid 和 tanh 更快

（3）被认为具有生物学合理性（Biological Plausibility），比如单侧抑制、宽兴奋边界（即兴奋程度可以非常高）

ReLU缺点：

（1）当输入为负时，ReLU 完全失效，在正向传播过程中，这不是问题。有些区域很敏感，有些则不敏感。但是在反向传播过程中，如果输入负数，则梯度将完全为零

（2）不以零为中心：和 Sigmoid 激活函数类似，ReLU 函数的输出不以零为中心，ReLU 函数的输出为 0 或正数，给后一层的神经网络引入偏置偏移，会影响梯度下降的效率

### 🌻2.4 SiLU  

SiLU激活函数（又称Sigmoid-weighted Linear Unit）是一种新型的非线性激活函数，它是Sigmoid和ReLU的改进版。SiLU具备无上界有下界、平滑、非单调的特性。SiLU在深层模型上的效果优于 ReLU。可以看做是平滑的ReLU激活函数。SiLU激活函数的特征是它在低数值区域中表现较为平滑，而在高数值区域中表现起来十分“锐利”，从而能够有效地避免过度学习的问题。

函数的表达式如下：

![](https://i-blog.csdnimg.cn/blog_migrate/a64ae1a6f5772a0af2e491ab8019fc1f.png)

图像如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8e3be6f40a6e939ee22ba0f347d75708.gif)

SiLU优点：

（1）相对于ReLU函数，SiLU函数在接近零时具有更平滑的曲线，并且由于其使用了sigmoid函数，可以使网络的输出范围在0和1之间。

SiLU缺点：

（1）引入了指数函数，增加了计算量。

### 🌻2.5 swish激活函数 

swish 是一个平滑的、非单调的函数，在深度网络上始终匹配或优于 ReLU，适用于各种具有挑战性的领域，如图像分类和机器翻译。通过self-gating，它只需要一个标量输入，而在多选通场景中，它需要多个双标量输入。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/38784dfff8a52785e50cdeb7627be0d6.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/02c7c57fff29fa2ec2bbfa4d96c51fb4.png)

swish优点：

（1）无上界（避免过拟合）

（2）有下界（产生更强的正则化效果）

（3）平滑（处处可导 更容易训练）

（4）x<0具有非单调性（对分布有重要意义 这点也是Swish和ReLU的最大区别）

### 🌻2.6 hardswish激活函数 

hardswish激活函数是对swish激活函数的改进，尽管swish非线性激活函数提高了检测精度，但不适合在嵌入式移动设备上使用，因为“S”型函数在嵌入式移动设备上的计算成本更高，求导较为复杂，在量化时计算较慢。但是如果在实验中使用hardswish非线性激活函数在准确性没差别的情况下部署在嵌入式移动设备上却会具有多重优势。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/4d7c8e3117f01fe4a765e9fc556b4293.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/07fe00cb49bf8aa68b53c1071ff93bdb.png#pic_center)

hardswish优点：

（1）hardswish相较于swish函数，具有数值稳定性好，计算速度快等优点

（2）消除了由于近似Sigmoid形的不同实现而导致的潜在数值精度损失

（3）在实践中，hardswish激活函数可以实现为分段功能，以减少内存访问次数，从而大大降低了等待时间成本

### 🌻2.7 Mish激活函数 

Mish一种新型的自正则化的非单调激活函数，拥有无上界(unbounded above)、有下界(bounded below)、平滑(smooth)和非单调(nonmonotonic)四个优点。其在ImageNet上的效果比ReLU和Swish都要好。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/160a28094f106d52d3d97e59972da3ea.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/ff0969f1f5c111aea6ffbd74b49b4a62.png)

Mish优点：

（1）没有上限

（2）有下限

（3）非单调性

（4）无穷阶连续性和光滑性

### 🌻2.8 ELU激活函数 

ELU（Exponential Linear Unit）激活函数是一种添加了指数项的非线性激活函数，ELU 激活函数的特征在于输入为负时会有正值的输出，而不是 0。这样可以有效地避免神经元死亡问题。 ELU 激活函数有助于神经元学习复杂的非线性函数，可以帮助神经元学习复杂的非线性特征。  


函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/d05dc88f0acb2785416d01d94ca9b673.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/db9da4114256737ea93e41b65d9e9b91.png)

ELU优点：

（1）不会有Dead ReLU问题

（2）输出的均值接近0。 能得到负值输出，这能帮助网络向正确的方向推动权重和偏置变化

ELU缺点：

（1）由于包含指数运算，所以计算量稍大

（2）无法避免梯度爆炸问题

（3）神经网络不学习 α 值

### 🌻2.9 AconC/MetaAconC激活函数 

这是2021年新出的一个激活函数，先从ReLU函数出发，采用Smoth maximum近似平滑公式证明了Swish就是ReLU函数的近似平滑表示，这也算提出一种新颖的Swish函数解释（Swish不再是一个黑盒）。之后进一步分析ReLU的一般形式Maxout系列激活函数，再次利用Smoth maximum将Maxout系列扩展得到简单且有效的ACON系列激活函数：ACON-A、ACON-B、ACON-C。最终提出meta-ACON，动态的学习（自适应）激活函数的线性/非线性，显著提高了表现。

函数的表达式如下：  
![](https://i-blog.csdnimg.cn/blog_migrate/27aebeaf3c3431cc93a468ba18526bb6.png#pic_center)

图像如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2a2c3fe5ab50ea3e55d9df00d826fbb7.png#pic_center)

AconC/MetaAconC优点：

（1）动态的学习（自适应）激活函数的线性/非线性，显著提高了表现

（2）ACON的参数P1和P2负责控制函数的上下限，也就是允许神经元自适应激活或者不激活，这个动态控制函数线性与非线性能力的参数就可以进行一个自适应学习

### 🌻2.10 Softplus激活函数 

Softplus函数可以看作是ReLU函数的平滑。根据神经科学家的相关研究，Softplus函数和ReLU函数与脑神经元激活频率函数有神似的地方。也就是说，相比于早期的激活函数，Softplus函数和ReLU函数更加接近脑神经元的激活模型，而神经网络正是基于脑神经科学发展而来，这两个激活函数的应用促成了神经网络研究的新浪潮。

函数的表达式如下：![](https://i-blog.csdnimg.cn/blog_migrate/a6cf6c24e3e1ebed210b6d35d2cd08d2.png)

图像如下：

![](https://i-blog.csdnimg.cn/blog_migrate/913c7ba025d14e6ee0456ed575d6958c.png)

Softplus优点：

（1）对于有些输出，需要更加平滑和连续性，而 Softplus相对于Relu，是一种更加平滑的激活函数

Softplus缺点：

（1）计算复杂，涉及到e的指数计算，计算量大

（2）这个函数对于硬件化也不友好

## 🌲三、更换激活函数的步骤 

#### 第①步 打开activations.py文件，查看已有函数 

首先我们找到utils/activations.py文件，位置如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/85fceb8d01b5e0621d0672af42338ebd.png)

打开这个文件，我们可以发现，SiLU，Hardswish，Mish，MemoryEfficientMish，FReLU，AconC，MetaAconC 已经在文件里定义了，这是我们可以直接用的。

![](https://i-blog.csdnimg.cn/blog_migrate/4d26a48523b5325de84e54fcb2e63c36.png)

#### 第②步 在common.py中更换激活函数  

先导入要更换的函数，这里以MetaAconC为例：

```java
from utils.activations import MetaAconC
```

然后再在相应位置修改，如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/c9f2c1b365d782698fc3162079afd58d.png)

如果想更换其他激活函数，也可以参照下面代码，步骤都是一样的：

```java
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Identity() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Tanh() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Sigmoid() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = AconC(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = MetaAconC(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = SiLU_beta(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.act = FReLU_noBN_biasFalse(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = FReLU_noBN_biasTrue(c2) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

## 🌲四、如何选择合适的激活函数 

其实关于激活函数的选择是调参中的玄学，不同的数据集效果不同，以下是我参考一些资料总结的一点经验，也欢迎大家在评论区补充噢！

> 1.  小白可以从ReLU函数开始，如果ReLU函数没有提供最优结果，再尝试其他激活函数。
> 2.  如果使用 ReLU，那么一定要小心设置 learning rate，而且要注意不要让网络出现很多 dead神经元，如果出现这个问题，那么可以试试 Leaky ReLU、PReLU 或者 Maxout（貌似PReLU效果最好）。
> 3.  ReLU函数只能在隐藏层中使用。
> 4.  在搭建神经网络时，如果搭建的神经网络层数不多，选择 Sigmoid、Tanh、ReLU，LeakyReLU，Softmax 都可以；而如果搭建的网络层次较多，那就需要小心，选择不当就可导致梯度消失问题。
> 5.  由于梯度消失问题，有时要避免使用sigmoid和tanh函数。
> 6.  用于分类器时，Sigmoid函数及其组合通常效果更好。（除非在二分类问题中，否则请小心使用Sigmoid函数）

> 本文参考：
> 
> [深度学习笔记：如何理解激活函数？（附常用激活函数） - 知乎 (zhihu.com)][- _ _zhihu.com]
> 
> [YOLOv5/v7 更换激活函数\_yolov5修改激活函数\_迪菲赫尔曼的博客-CSDN博客][YOLOv5_v7 _yolov5_-CSDN]

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

![](https://i-blog.csdnimg.cn/blog_migrate/9399627be98e63e912c64c82bd1ccf13.gif)


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
[Link 1]: #%F0%9F%8C%B2%E4%B8%80%E3%80%81%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E7%AE%80%E4%BB%8B
[1.1]: #%F0%9F%8C%BB1.1%20%E4%BB%80%E4%B9%88%E6%98%AF%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%EF%BC%9F
[1.2]: #%F0%9F%8C%BB1.2%20%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E6%9C%89%E4%BB%80%E4%B9%88%E7%94%A8%EF%BC%9F
[Link 2]: #%F0%9F%8C%B2%E4%BA%8C%E3%80%81%E4%B8%80%E4%BA%9B%E5%B8%B8%E8%A7%81%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.1 sigmoid]: #%F0%9F%8C%BB2.1%20sigmoid%E5%87%BD%E6%95%B0
[2.2 tanh _]: #%F0%9F%8C%BB2.2%20tanh%20%2F%20%E5%8F%8C%E6%9B%B2%E6%AD%A3%E5%88%87%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.3 ReLU]: #%F0%9F%8C%BB2.3%C2%A0ReLU%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.4 SiLU]: #%F0%9F%8C%BB2.4%20SiLU%C2%A0
[2.5 swish]: #%F0%9F%8C%BB2.5%20swish%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.6 hardswish]: #%F0%9F%8C%BB2.6%C2%A0hardswish%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.7 Mish]: #%F0%9F%8C%BB2.7%C2%A0Mish%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.8 ELU]: #%F0%9F%8C%BB2.8%C2%A0ELU%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.9 AconC_MetaAconC]: #%F0%9F%8C%BB2.9%C2%A0AconC%2FMetaAconC%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.10 Softplus]: #%F0%9F%8C%BB2.10%C2%A0Softplus%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[Link 3]: #%F0%9F%8C%B2%E4%B8%89%E3%80%81%E6%9B%B4%E6%8D%A2%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E6%AD%A5%E9%AA%A4
[_activations.py]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%20%E6%89%93%E5%BC%80activations.py%E6%96%87%E4%BB%B6%EF%BC%8C%E6%9F%A5%E7%9C%8B%E5%B7%B2%E6%9C%89%E5%87%BD%E6%95%B0
[_common.py_]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%20%E5%9C%A8common.py%E4%B8%AD%E6%9B%B4%E6%8D%A2%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%C2%A0
[Link 4]: #%F0%9F%8C%B2%E5%9B%9B%E3%80%81%E5%A6%82%E4%BD%95%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[- _ _zhihu.com]: https://zhuanlan.zhihu.com/p/364620596
[YOLOv5_v7 _yolov5_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/124413941?spm=1001.2014.3001.5502
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