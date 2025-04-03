## ![](https://i-blog.csdnimg.cn/blog_migrate/61d57cd84282ef4f532f6859b3acbde9.jpeg) 

## 前言 

今天我们一起来学习何恺明大神的又一经典之作： ResNeXt（《Aggregated Residual Transformations for Deep Neural Networks》）。这个网络可以被解释为 VGG、ResNet 和 Inception 的结合体，它通过重复多个block（如在 VGG 中）块组成，每个block块聚合了多种转换（如 Inception），同时考虑到跨层连接（来自 ResNet）。

在学习之前，我们先回顾一下这三个经典网络：

VGG：[经典神经网络论文超详细解读（二）——VGGNet学习笔记（翻译＋精读）][VGGNet]

Inception：[经典神经网络论文超详细解读（三）——GoogLeNet InceptionV1学习笔记（翻译＋精读+代码复现）][GoogLeNet InceptionV1]

ResNet：[经典神经网络论文超详细解读（五）——ResNet（残差网络）学习笔记（翻译＋精读＋代码复现）][ResNet]

原文地址：[Aggregated Residual Transformations for Deep Neural Networks (arxiv.org)][Aggregated Residual Transformations for Deep Neural Networks _arxiv.org]

代码地址： [GitHub - facebookresearch/ResNeXt: Implementation of a classification framework from the paper Aggregated Residual Transformations for Deep Neural Networks][GitHub - facebookresearch_ResNeXt_ Implementation of a classification framework from the paper Aggregated Residual Transformations for Deep Neural Networks]

## 目录 

[前言][Link 1]

[Abstract—摘要][Abstract]

[一、Introduction—简介][Introduction]

[二、Related Work—相关工作][Related Work]

[2.1Multi-branch convolutional networks—多分支卷积网络][2.1Multi-branch convolutional networks]

[2.2Grouped convolutions—分组卷积][2.2Grouped convolutions]

[2.3Compressing convolutional networks—压缩卷积网络][2.3Compressing convolutional networks]

[2.4Ensembling—集成][2.4Ensembling]

[三、Method—方法][Method]

[3.1. Template—模型][3.1. Template]

[3.2. Revisiting Simple Neurons—回顾单个神经元][3.2. Revisiting Simple Neurons]

[3.3. Aggregated Transformations—聚合变换][3.3. Aggregated Transformations]

[3.4. Model Capacity—模型能力][3.4. Model Capacity]

[四、Implementation details—执行细节][Implementation details]

[五、Experiments—实验][Experiments]

[5.1. Experiments on ImageNet-1K—在ImageNet-1K的实验][5.1. Experiments on ImageNet-1K_ImageNet-1K]

[5.2. Experiments on ImageNet-5K—在ImgaeNet-5K的实验][5.2. Experiments on ImageNet-5K_ImgaeNet-5K]

[5.3. Experiments on CIFAR—在CIFAR的实验][5.3. Experiments on CIFAR_CIFAR]

[5.4. Experiments on COCO object detection—在COCO目标检测的实验][5.4. Experiments on COCO object detection_COCO]

[论文十问][Link 2]

## Abstract—摘要 

### 翻译 

我们提出了一种简单的，高度模块化的网络结构用于图像分类。我们的网络是通过堆叠Building Block进行构建的，这种Building Block聚合了一组具有相同拓扑的转换。我们简单的设计使得一个同构的、多分枝的体系结构只需要设置几个超参数。这种策略暴露了一个新的维度，我们称之为「基数」（转换集的大小），它是深度和宽度之外的另一个基本要素。在ImageNet-1K数据集上，我们的实验表明，即使在保持复杂度限制的条件下，增加基线也能够提高分类精度。另外，当我们的模型容量增加的时候，增加基线比增加宽度或高度更加有效。我们把模型命名为 ResNeXt，这个模型带领我们进入了ILSVRC 2016分类任务，并获得了第二名的成绩。我们进一步研究了在ImageNet-5K数据集和COCO数据集上的ResNeXt性能表现，也显示出比ResNet更好的效果。代码和模型将在网络上公开访问。

### 精读 

#### 主要内容 

本文工作： 我们提出了一种用于图像分类的简单、高度模块化的网络架构。该网络通过反复堆叠Building Block实现，Building Block则通过聚集简洁的卷积模块来实现。

本文方法： 本文提出一个与网络宽度和深度类似作用的参数，用来衡量网络大小，称之为Cardinality基数（转换集的大小）。

本文优点： 该网络具有相同的、多分支的结构，并且对应的超参数非常少。

本文结论：

（1）ImageNet-1K数据集上，不增加模型复杂度，增加Cardinality可以提升网络性能。

（2）增加Cardinality比增加网络深度和宽度可以更好的提升网络模型的性能。

本文成果： ILSVRC 2016分类任务获得了第二名。

## 一、Introduction—简介 

### 翻译 

视觉识别的研究正在从“特征工程”到“网络工程”\[25,24,44,34,36,38,14\]的过渡。与传统的手指特征（例如，SIFT \[29\]和HOG \[5\]）相比，由大规模数据的神经网络学习的特征需要在训练期间需要最小的人类参与，并且可以转移到各种识别任务\[7,10,28\]。尽管如此，人类努力已经转移到设计更好的网络架构以获得学习陈述。

设计架构越来越困难，越来越多的超参数（宽度2，过滤尺寸，步幅等），尤其是当有许多层时。 VGG-Net \[36\]展示了构建非常深网络的简单而有效的策略：堆叠相同shape的组件。该策略继承于Resnet\[14\]，Resnet堆叠了相同拓扑的模块。这个简单的规则减少了超参数的自由选择，深度揭示了神经网络中的基本维度。此外，我们认为这条规则的简单性可能会降低过度调整超参数到特定数据集的风险。通过各种视觉识别任务\[7,10,9,28,31,14\]并通过涉及语音\[42,30\]和语言\[4,41,20\]的非视觉任务已经被证明的VGGNET和RESNET的鲁棒性。  
与VGG-nets不同，Inception模型系列\[38,17,39,37\]已经证明，精心设计的有着较低理论复杂性的拓扑结构足以实现令人信服的精确率。随着时间的推移，Inception模型已经进化\[38,39\]，但重要的常见属性是分裂变换合并策略。在Inception模块中，输入被分成几维低维的输入（按1×1卷积），由一组专用过滤器（3×3，5×5等）转换，并通过连接合并。可以表明，这种架构的解决方案是在高维输入的单个大网络层（例如，5×5）的解空间的严格子空间。预计Inception模块的分流变换合并行为将接近大而致密层的代表性，但是具有相当较低的计算复杂性。  
尽管准确性良好，但Inception模型的实现已经伴随着一系列复杂的因素——过滤器数量和大小是为每个单独的变换定制的并且模块是分阶段自定义的。虽然这些组件的仔细组合产生了优秀的神经网络结构，但通常不清楚如何将初始化架构调整到新的数据集/任务，特别是当有许多因素和超参数时要设计。  
在本文中，我们提出了一种简单的架构，它采用VGG / Resnets的重复层策略，同时以简单，可扩展的方式利用分流交换合并策略。我们网络中的模块执行一组变换，每执行一个低维输入时，其输出通过求和聚合。我们追求这种想法的简单实现 - 要汇总的转换是所有相同的拓扑（例如，图1（右））。这种设计允许我们扩展到任何大量的转换，无需专门设计。我们追求这种想法的简单实现-要汇总的变换是所有相同的拓扑（例如，图1（右））。这种设计使我们能够扩展到没有专门设计的大量变换。  
有趣的是，根据这种简化，我们的模型具有另外两种等同形式（图3）。图3（b）中的重构类似于Inception-ResNet模块\[37\]，其中它连接多个路径;但是我们的模块与所有现有的Inception模块不同，因为我们所有路径都共享相同的拓扑，因此可以容易地隔离路径的数量作为要调查的因素。在更加简洁的重构中，我们的模块可以由Krizhevsky等人重塑。的分组卷积\[24\]（图3（c）），然而，这一点被制定为工程妥协。  
我们经验证明，即使在维持计算复杂度和模型大小的限制条件下，我们的聚合变换也优于原始的Reset模块 - 例如，图1（右）旨在保持图1的浮点复杂性和参数的参数数量。1（剩下）。我们强调，虽然通过增加容量（进展更深或更宽）的方法相对容易地提高准确性，但在维持（或降低）复杂性的同时提高精度的方法是罕见的。  
我们的方法表明，除了宽度和深度之外，基数(变换集合的大小)是一个具有中心重要性的具体的、可测量的维度。实验表明，增加基数是一种比增加深度或增加宽度更有效的获得准确性的方法，特别是当深度和宽度开始为现有模型带来递减的回报时。  
我们的神经网络，命名为ResNeXt(暗示下一个维度)，在ImageNet分类数据集上优于ResNet-101/152\[14\]、ResNet200\[15\]、Inception-v3\[39\]和Inception- ResNet -v2\[37\]。特别是，101层的ResNeXt能够达到比resnet - 200\[15\]更好的精度，但复杂度只有50%。此外，ResNeXt展示的设计比所有Inception模型都要简单得多  
ResNeXt是我们提交ILSVRC 2016分类任务的基础，我们在该任务中获得了第二名。本文进一步评估了ResNeXtona larger imagenet-5kset和COCO object检测数据集\[27\]，始终显示出比ResNet同类数据集更好的准确性。我们希望ResNeXt也能很好地推广到其他视觉(和非视觉)识别任务。

### 精读 

#### 之前方法 

深度研究

VGG: 结构简洁， 堆叠使用3\*3卷积。

不足：但是随着层数增多，出现梯度消失问题

ResNet： 堆叠的Building Block采用残差结构，用跨层连接缓解了这个问题，使得网络可以达到上千层。

不足：但是模型的参数量巨大

宽度研究

Inception系列：采用多分支结构Split- Transform-Merge（分割-变换-聚合）

1） Split：将向量x分成低维嵌入表示；（由1x1卷积降维）

2） Transform：每个低维特征经过一个线性变换；（再由3x3或者5x5的卷积进一步提取特征）

3） Merge：通过单位加合成最后的输出；（最后拼接各分支的特征）

不足：但是每个映射变换要量身定制卷积核数量、尺寸，模块在每一阶段都要改变。尤其将 Inception 模型用于新的数据或者任务时如何修改并不清晰。

#### 本文改进 

本文提出了一个简单架构，采用 VGG/ResNets 重复相同网络层的策略，以一种简单可扩展的方式延续Split-Transform-Merge 策略。将ResNet中高维特征图分组为多个相同的低维特征图，然后在卷积操作之后，将多组结构进行求和，最后得到ResNeXt模型。

## 二、Related Work—相关工作 

### 翻译 

多分支卷积网络。Inception模型\[38,17,39,37\]是成功的多分支架构，其中每个分支经过仔细定制。 ResNet \[14\]可以被认为是一个分支网络，其中一个分支是恒等映射。深度神经决策森林\[22\]是具有学习拆分功能的树模式的多分支网络。  
分组卷积。使用分组的卷积可以追溯到AlexNet论文\[24\]，如果不是之前Krizhevsky等人提供的想法 \[24\]用于将模型分布在两个GPU上。 Caffe \[19\]，Torch\[3\]和其他框架支持分组的卷积，主要用于AlexNet的兼容性。据我们所知，有关利用分组卷积的一点证据，以提高准确性。分组卷积的特殊情况是明智通道的卷积，其中组的数量等于渠道数量。明智通道的卷积是\[35\]中可分离卷积的一部分。  
压缩卷积网络。分解（在空间\[6,18\]和/或通道\[6,21,16\]水平）是一种广泛采用的技术，以减少深卷积网络的冗余并加速/压缩它们。 Ioannou等人\[16\]呈现用于减少计算的“根”-模式网络，并且通过分组的卷积来实现根部的分支。这些方法\[6,18,21,16\]显示了具有较低复杂性和更小的模型尺寸的精度的优雅妥协。我们的方法代替压缩，而不是压缩，这是一个经验呈现更强大的代表性的架构。  
集合。平均一组独立培训的网络是提高准确性的有效解决方案\[24\]，广泛采用识别竞争\[33\]。 VEit等人 \[40\]将单个RESET解释为较浅网络的集合，从ResNet的想加表现导致了\[15\]。我们的方法利用添加了一组变换。但是，我们认为它是不精确的，以便将我们的方法视为合奏，因为要聚合的成员是联合训练，而不是独立训练的。

### 精读 

（1）ResNets可以被认为是两个分支网络，其中一个分支是identity 映射。深度神经决策森林是具有学习分裂函数的树模式多分支网络；深度网络决策树也是多分支结构。

（2）几乎没有证据表明分组卷积可提升网络性能。

（3）本文不同于模型压缩，本文设计的结构自身就有很强的性能和低计算量。

（4）模型集成是有效的提高精度的方法；本文模型并不是模型集成，因为各模块的训练是一起同时训练的，并不是独立的。

## 三、Method—方法 

### 3.1. Template—模型 

### 翻译 

我们采用高度模块化的设计，遵循VGG/ResNets.我们的网络由堆叠的残差块组成。这些块具有相同的拓扑，并受到VGG / Resnets的两个简单规则的影响：（i）如果产生相同大小的空间映射，则该块共享相同的超参数（宽度和滤波器大小），以及（II ）每次当空间映射下采样因子为2时，块的宽度就乘以2。第二条规则确保计算复杂性，按照浮点操作（浮点操作中，在＃中乘法添加）对于所有块大致相同。  
通过这两个规则，我们只需要设计模板模块，并且可以相应地确定网络中的所有模块。因此，这两个规则极大地缩小了设计空间，让我们专注于一些关键因素。由这些规则构建的网络在表1中。

![](https://i-blog.csdnimg.cn/blog_migrate/26885748ede135788e8c15bb50b78c95.png)

### 精读 

#### 模型设计两个原则： 

（1）如果输出的空间尺寸一样，那么模块的超参数（宽度和卷积核尺寸）也是一样的。

（2）每当空间分辨率/2（降采样），则卷积核的宽度\*2。这样保持模块计算复杂度。

#### 作用： 

这两条规则大大缩小了设计空间，让我们可以专注于几个关键因素。

![](https://i-blog.csdnimg.cn/blog_migrate/c4f8a52197f19e988c3ff92399dd114c.png)

> 32 指进入网络的第一个ResNeXt基本结构的分组数量C（即基数）为32
> 
> 4d 表示depth即每一个分组的通道数为4（所以第一个基本结构输入通道数为128）

可以看到ResNet-50和ResNeXt-50（32x4d）拥有相同的参数，但是精度却更高。

### 3.2. Revisiting Simple Neurons—回顾单个神经元 

### 翻译 

人工神经网络中最简单的神经元执行内积(加权和)，这是由全连通和卷积层完成的初等变换。内积可以看作是一种聚集变换形式:

![](https://i-blog.csdnimg.cn/blog_migrate/9c8eb052822d9fbdd33bd0fb3665fa6a.png)

其中x = \[ x1 , x2 , … , xD \]是神经元的一个具有D通道的输入向量，并且wi 是一个卷积核第i通道权重。这个操作(通常包括一些输出非线性)被称为“神经元”。参见图2。

![](https://i-blog.csdnimg.cn/blog_migrate/8db328a07feefe921d4e2b4de0623b07.png)

可以将上述操作理解为分裂、变换和聚合的组合。  
(i) 分割:将向量x切为低维输入，其中为单维子空间xi  
(ii)变换：对低维表示进行变换，并对其进行简单缩放：wixi  
(iii)聚合：所有输入的变换由公式（1）聚合

### 精读 

#### 单个神经元 

单个神经元是构建全链接和卷积网络层的基础元素。

具体操作：

分割（Splitting）： 把输入x分解为D个元素，可以理解为低维嵌入。

变换（Transforming）： 每个元素进行变换，乘权重wi进行缩放。

聚合（Aggregating）： 对D个变换后的结果进行求和。

### 3.3. Aggregated Transformations—聚合变换 

### 翻译 

考虑到以上对单个神经元的分析，我们考虑用更一般的函数代替初等变换（wixi），它本身也可以是一个网络。与原来增加深度维度的“网络中的网络”\[26\]相比，我们的“神经元中的网络”沿着一个新的维度扩展。  
形式上，我们将聚合转换表示为

![](https://i-blog.csdnimg.cn/blog_migrate/b272f1c5fb9df0865710007c4fd1f497.png)

其中Ti(x) 可以是任意函数。类似一个简单的神经元，Ti 应该将x 投影到一个(可选的低维)输入,然后将其变换在Eqn.(2)中，C 是要聚合的转换集合的大小。我们把C 称为基数\[2\]。在Eqn.(2)中C 的位置与Eqn.(1)中的D 相似，但C 不必等于D ，可以是一个任意的数。虽然宽度的维度与简单转换(内积)的数量有关，但我们认为基数的维度控制更复杂转换的数量。我们通过实验表明基数是一个基本的维度，可以比宽度和深度的维度更有效。  
在本文中，我们考虑了一种简单的设计变换函数的方法:所有的Ti 具有相同的拓扑。这扩展了VGG风格的策略，即重复相同形状的层，这有助于隔离一些因素并扩展到任何大量的转换。我们将个别的转换Ti设置为瓶颈型架构\[14\]，如图1(右)所示。在本例中，每个Ti中的第一个1×1层产生低维输入 ![](https://i-blog.csdnimg.cn/blog_migrate/bd38cae7e70d8699d45b86ce4e8edfed.png)

图3.等价的ResNext的构建块。  
（a）：聚合的残差变换，与图1右部分相同  
（b）：相当于(a)的一个块，实现为提前串联  
（c）：等价于(a,b)的一个块，实现为分组卷积  
粗体文本的符号将突出显示重新构建的改变。一个层表示为（input channels，filter size，output channels)  
Eqn.(2)中的聚合变换作为残差函数(图1右):

![](https://i-blog.csdnimg.cn/blog_migrate/6c82e508d3d0f9ddae7c211e34e9b437.png)

其中y 是输出。  
与Inception-ResNet的关系。  
一些张量处理表明，图1(右)中的模块(图3(a)中也展示了)等价于图3(b)。图3(b)与Inception-ResNet\[37\]的块相似，它在残差函数中涉及分支和串联。但与所有Inception或Inception- ResNet模块不同的是，我们在多个路径之间共享相同的拓扑结构。我们的模块需要最少的额外努力设计每个路径。  
与分组卷积的关系。使用分组卷积\[24\].4的表示法，上面的模块变得更加简洁。图3©说明了这种重新构建的结构。所有的低维输入(第一个1×1层)都可以用一个更宽的单一层代替(如图3©中的1×1, 128-d)。分裂本质上是由分组卷积层完成的，当它把输入通道分成组时。图3(c)中的分组卷积层进行了32组卷积，其输入和输出通道为四维。（这里将输入128通道分为32组，每组4个通道）  
分组的卷积层将它们串联起来作为层的输出。图3©中的区块与图1(左)中的原始瓶颈剩余区块相似，但图3©是一个较宽但连接稀疏的模块。

![](https://i-blog.csdnimg.cn/blog_migrate/7244f1a714bb008f7a4461233b5e4d25.png)

上图左右两边的结构是等价的。

### 精读 

#### Block中的聚合变换 

公式表示

对于一个ResNeXt Block中的基数块输出可以表示为：

![](https://i-blog.csdnimg.cn/blog_migrate/500b1b5c6b7f5d01184977d264ff86cd.png)

其中，参数C 代表基数块的数目，Ti 代表对应的基数块，将x投影到一个(可选的低维)集成中，然后对其进行变换。

这拓展了VGG设计原则：从重复相同大小的层，到重复相同拓扑的卷积核组。

在本文中，我们考虑一种设计变换函数的简单方法：所有的Ti都有相同的拓扑结构。

在这种情况下，每个Ti中的第一个1×1层产生低维嵌入。那么对应的残差网络输出就可以被表示为：

![](https://i-blog.csdnimg.cn/blog_migrate/5fa3d1e8a76455f2f618c46570e0fc01.png)

图像表示

![](https://i-blog.csdnimg.cn/blog_migrate/bbe64bfbb6b582512c505f3dba886759.jpeg)

具体操作

Splitting： 通过1×1卷积实现低维嵌入，256个通道变成4个通道，总共32个分支（cardinality = 32）

Transforming： 每个分支进行变换（对网络层对数据操作）

Aggregating： 对32个分支得到的变换结果—特征图，进行聚合

#### Block的三种等效形式 

（a）表示先划分，单独卷积并计算输出，最后输出相加。split-transform-merge三阶段形式

（b）表示先划分，单独卷积，然后拼接再计算输出。将各分支的最后一个1×1卷积聚合成一个卷积。

（c）就是分组卷积。将各分支的第一个1×1卷积融合成一个卷积，3×3卷积采用group（分组）卷积的形式，分组数=cardinality（基数）

（c）结构分析

（1）首先通过一个1x1的卷积层进行降维处理，将它的channel从256降低到128,

（2）然后在通过group卷积进行处理，这里group 卷积的卷积核为3x3它的groups数为32，它所输出的channel也是等于128,

（3）接着通过1x1卷积对它进行升维

（4）最后将它的输出与我们的输入进行相加得到最后输出。

> #### 三个结构为什么等价？ 
> 
> （1）b和c等价
> 
> 第一层：
> 
> 过程：首先从(b)到(c)这个过程，对于(b)中第一层通过包括32个分支，每个分支(path)卷积核个数为4的1x1卷积，对于每个path而言它的卷积核大小都是1x1，channel为256，又由于我们path的个数为32，就可以简单将他们合并在一起，变为( c)图中第一层了。
> 
> 参数：
> 
> （b）第一层 256×1×1×4×32=32768
> 
> （c）第一层 256×1×1×128=32768
> 
> 第二层：
> 
> 过程：和group卷积其实是一样的，对于每个path可以理解为一个group，每个组的输入输出channel为原来的1/group,对于每个组采用3x3的卷积核，卷积之后将特征矩阵进行concate拼接，所以图(b)第二层也是与图(c)第二层 group为32的组卷积也是等价的。
> 
> 参数：
> 
> （b）第二层 4×3×3×4×32=4608
> 
> （c）第二层 128/32×3×3×128 = 4608
> 
> （2）a和b等价
> 
> 过程：在（a）中，4维特征图通过1x1卷积变为256维，然后32个256维数据求和，而在(b)中，是先将4维数据concat成128维，在利用1x1卷积，实际上也就是求和过程。
> 
> 参数：
> 
> （a）第三层 4×1×1×256×32=32768
> 
> （b）第三层 128×1×1×256 = 32768

####  分组卷积 

在AlexNet时就曾提出，由受限于当时硬件的限制，作者不得不将卷积操作拆分到两台GPU上运行，这两台GPU的参数是不共享的。 两组卷积核学习两种不同的特征，一组学习纹理，另一组学习色彩。

![](https://i-blog.csdnimg.cn/blog_migrate/e7f553887d121599312fe46e566b4831.png)

操作：

分组卷积层中，输入和输出的 channels 被分为 C 个 groups，分别对每个 group 进行卷积操作。

优点：

（1）减少参数量，分成G组，则该层的参数量减为原来的1/G。

![](https://i-blog.csdnimg.cn/blog_migrate/52de29b7045ba3f950275ad6c4f32d80.jpeg)

（2）让网络学习到不同的特征，每组卷积学习到的特征不一样，获得更丰富的信息。

（3）分组卷积可以看做是对原来的特征图进行了一个dropout，有正则的效果。

####  讨论 

对于基数块中的深度（depth）论文中做了讨论

![](https://i-blog.csdnimg.cn/blog_migrate/476cc73e7663b9edfcaac9fd13a95f89.png)

 结论：

如上图所示，如果一个ResNeXt Block中只有两层conv，前后都可等效成一个大的conv层，那聚合变换和没聚一样。

### 3.4. Model Capacity—模型能力 

### 翻译 

我们在下一节中的实验将表明，在保持模型复杂性和参数数量的情况下，我们的模型可以提高精度。这不仅在实践中很有趣，更重要的是，参数的复杂性和数量代表了模型的固有容量，因此通常被研究为深层网络的基本属性\[8\]。  
当我们在保持复杂性的同时评估不同的基数时，我们希望最小化对其他超参数的修改。我们选择去适配BottleNeck的宽度（图1右边的4-d），因为它可以与模块的输入和输出隔离。这种策略对其他超参数(块的深度或输入/输出宽度)没有引入任何变化，因此有助于我们关注基数的影响。

在图1（左），原始的ResNet BottleNeck Block\[14\]有25664+336464+64\*256=70K的参数并且成正比的浮点数（在相同的特征图大小上）。对于BottleNeck宽度d，我们的模板在图1有：

![](https://i-blog.csdnimg.cn/blog_migrate/b65a2efe1c2d638708440deefab35efc.png)

当C=32和d=4时,等式(4)约等于70K。表2显示了基数C和瓶颈宽度d 之间的关系。

![](https://i-blog.csdnimg.cn/blog_migrate/77d528c5bcd3cc7bacc8e44a22e43e67.png)

因为我们采用3.1节的两个规则，上述近似等式在ResNet的BottleNeck Block和我们的ResNext之间的所有阶段都有效(除了子采样层，其中特征图大小发生变化)。表1比较了原始的ResNet-50和我们容量相似的ResNext-50。我们注意到复杂性只能近似保留，但复杂性的差异很小，不会影响我们的结果。

### 精读 

两个重要参数： Cardinality和Width

结论：

（1）C逐渐增大时，模型性能的变化；

（2）设计C和d，使得整体的参数量在70k左右。目的是对标ResNet。

## 四、Implementation details—执行细节 

### 翻译 

我们的实现遵循\[14\]和公开可用的代码fb.resnet.torch \[11\]。在ImageNet数据集上，使用\[11\]实现的\[38\]的比例和纵横比增强，从调整大小的图像中随机裁剪出224×224的输入图像。除了那些增加的维度是投影(在\[14\]中的类型B)，快捷方式是相同的连接。conv3、4和5的下采样是通过每个阶段第一个块的3×3层中的步长-2卷积来完成的，如\[11\]中所建议的。我们在8个图形处理器(每个图形处理器32个)上使用256个小批量的SGD。重量衰减为0.0001，动量为0.9。我们从0.1的学习率开始，用\[11\]中的计划表除以10，做三次。我们采用\[13\]的权重。在标记比较中，我们从一个短边为256的图像中评估单个224×224中心裁剪的误差  
我们的模型是通过图3©的形式实现的。我们执行了BN标准化\[17\]在图3（c）的卷积之后。ReLU在每个BN之后立即执行，除了在添加到快捷方式之后执行ReLU的块的输出，如下\[14\]。我们注意到，当BN和ReLU如上所述被适当地处理时，图3中的三种形式是严格等价的。我们训练了所有三种形式，得到了相同的结果。我们选择通过图3©来实现，因为它比其他两种形式更简洁、更快。

### 精读 

输入：从短边为 256 的图像中裁出一个 224x224 的图像进行测试

下采样： conv3、conv4、conv5 的下采样过程采用 stride 为 2 的 3x3 卷积。

处理器：使用 8 块 GPU 来训练模型

优化器：SGD

动量：0.9batch

尺寸：256 （每块 GPU 上 32）

重量衰减：0.0001

初始学习速率：0.1

实现：我们在卷积之后接BN，每个BN后加ReLU，除了最后输出的block（这里ReLU加在shortcut相加操作之后）。

## 五、Experiments—实验 

### 5.1. Experiments on ImageNet-1K—在ImageNet-1K的实验 

#### Notations—标记 

### 翻译 

我们在1000级ImageNet分类任务上进行消融实验\[33\]。我们按照\[14\]构造50层和101层残差网络。我们只是用我们的块替换ResNet-50/101中的所有块。  
标记。因为我们采用3.1节的两个规则，我们通过模板引用一个架构就足够了。例如，表1显示了由基数= 32、瓶颈宽度= 4d的模板构建的ResNeXt-50(图3)。为简单起见，该网络表示为ResNeXt-50 (32×4d)。我们注意到，模板的输入/输出宽度固定为256-d(图3)，每次对要素图进行二次采样时，所有宽度都会加倍(见表1)。

### 精读 

模板输入输出channel固定为256d，特征图大小/2时，channel×2。

####  Cardinality vs. Width—基数vs宽度 

### 翻译 

基数与宽度。我们首先评估基数C和bottleneck宽度之间的权衡，保留表2中列出的复杂性。表3显示了结果，图5显示了误差对时期的曲线。与ResNet-50(上表3和左图5)相比，32×4d ResNeXt-50的验证误差为22.2%，比ResNet基线的23.9%低1.7%。随着基数C从1增加到32，同时保持复杂度，错误率不断降低。此外，32×4d ResNeXt的训练误差也比ResNeXt低得多，这表明增益不是来自正则化，而是来自更强的表示。  
在ResNet-101的情况下也观察到类似的趋势(图5右侧，表3底部)，其中32×4d ResNeXt101的表现优于ResNet-101的0.8%。虽然这种验证误差的改善小于50层情况，但训练误差的改善仍然很大(ResNet-101为20%，32×4d ResNeXt-101为16%，图5右)。事实上，更多的训练数据将扩大验证误差的差距，正如我们在下一小节中在ImageNet-5K集上显示的那样。  
表3还表明，在保持复杂性的情况下，当bottleneck宽度小的时候以减少宽度为代价增加基数开始显示出饱和的准确性。我们认为，在这种情况下，继续缩小宽度是不值得的。所以我们在下面采用一个不小于4d的瓶颈宽度。

### 精读 

ResNet-50和ResNeXt-50比较

![](https://i-blog.csdnimg.cn/blog_migrate/4f7a2b9d3ddc6abb82d905cb9005b6af.png)

ResNet-101和ResNeXt-101比较

![](https://i-blog.csdnimg.cn/blog_migrate/4b5ccf98623c0342b80a8a5ee4fe1353.png)

结论：

（1）随着基数C的增加，复杂度相同的情况下，错误率持续降低，训练错误率也更低，这表明效果的提升不是来自正则化，而是来自更强的表示。

（2）在复杂度不变的情况下，提升基数，降低宽度所获得的效果提升逐渐饱和。因此，不需要再继续降低宽度了，我们之后使用等人宽度不低于4d。

#### Increasing Cardinality vs. Deeper/Wider—增加基数vs更深/更广 

### 翻译 

增加基数与更深/更宽。接下来，我们研究通过增加基数C或增加深度或宽度来增加复杂性。下面的比较也可以参考ResNet-101基线的2倍浮点运算来进行。我们比较了以下具有150亿次浮点运算的变体。(一)深入到200层。我们采用在\[11\]中实现的ResNet-200 \[15\]。（二）通过增加瓶颈宽度扩大范围。(三)通过加倍c来增加基数。  
表4显示，与ResNet-101基线(22.0%)相比，复杂度增加2倍会持续降低误差。但是当更深(ResNet200，0.3%)或更宽(更宽的ResNet-101，0.7%)时，改善很小。  
相反，相比增加深度或者宽度,增加基数C得到的结果更好。2×64d ResNeXt-101(即在1×64d ResNet-101基线上加倍并保持宽度)将top-1误差降低1.3%至20.7%。64×4d ResNeXt-101(即在32×4d ResNeXt-101上加倍C并保持宽度)将top-1误差降低到20.4%。  
我们还注意到，32×4d ResNet-101 (21.2%)比更深的ResNet-200和更宽的ResNet101性能更好，尽管它的复杂性只有50%。这再次表明基数比深度和宽度更有效。

### 精读 

下面我们研究通过增加基数或深度/宽度来提升模型复杂度的效果。

![](https://i-blog.csdnimg.cn/blog_migrate/1ebae9ca6682acb0c7eff01880744dcb.png)

第一个增加到200层；

第二个增加宽度；

第三个增加基数，C翻倍。

结论：

（1）表4中结果表明复杂度翻倍，错误率持续下降。

（2）但是使用更深层或更宽提升很小。

（3）相反，提升基数效果要好得多。

#### Residual connections—残差连接 

### 翻译 

残差连接。从ResNeXt-50中删除快捷方式会使错误增加3.9点，达到26.1%。从它的1497 ResNet-50对应物中删除快捷方式要糟糕得多(31.2%)。这些比较表明，剩余连接有助于优化，而聚合转换是更强的表示，这是因为它们始终比有或没有剩余连接的对应转换表现得更好。

### 精读 

下面这个表展示了残差连接的效果：

![](https://i-blog.csdnimg.cn/blog_migrate/bbf2202c5726b6eb3874359d754bfdee.png)

 结论：这些对比表示残差连接对于优化很有帮助，而我们的方法（变换聚合）有着更强的表示能力。

####  Performance—执行 

### 翻译 

执行。为了简单起见，我们使用Torch内置的分组卷积实现，没有特殊的优化。我们注意到这个实现是暴力的，并且不支持并行化。在NVIDIA M40的8个GPU上，表3中的32×4d ResNeXt-101训练每小批量需要0.95秒，而ResNet-101基线的0.70秒具有相似的浮点运算。我们认为这是合理的开销。我们期望精心设计的较低级别的实现(例如，在CUDA中)将减少这种开销。我们还期望CPU上的推理时间将呈现更少的开销。训练2×复杂性模型(64×4d ResNeXt-101)每个小批量需要1.7秒，在8个图形处理器上总共需要10天。

### 精读 

在8GPU上训练32 \* 4d ResNeXt-101每个minibatch花费0.95s，而对有着相当复杂度的ResNet-101 baseline，只需要0.7s。

#### Comparisons with state-of-the-art results—与最先进的结果进行比较 

### 翻译 

与最先进结果的比较。表5显示了ImageNet验证集的单作物测试的更多结果。除了测试224×224作物外，我们还评估了320×320 crop，如下所示\[15\]。我们的结果优于ResNet、Inception-v3/v4和Inception-ResNet-v2，single-crop前5名的错误率为4.4%。此外，我们的架构设计比所有的初始模型都简单得多，并且需要手动设置的超参数也少得多。  
Resnext是我们的参赛作品的基础，对ILSVRC 2016分类任务，我们实现了2个。我们注意到使用多尺度和/或多帧测试后，许多模型（包括我们的）开始在此数据集上饱和。使用\[14\]中的多尺度密集测试，我们的单模型top-1/top-5错误率为17.7%/3.7%，与采用多尺度多作物测试的Inception-ResNet-v2的单模型结果17.8%/3.7%相当。我们在测试集中进行了3.03％的前5个错误的集合结果，与winner的2.99％和Inception-V4 / InceptionResnet-V2的3.08％\[37\]相符。

###    精读 

表5展示了更多的单crop测试的结果对比。

![](https://i-blog.csdnimg.cn/blog_migrate/6f5a987eefe285402f6461d590f1412c.png)

结论：

（1）我们的结果优于ResNet、Inception-v3/v4和Inception-ResNet-v2，single-crop前5名的错误率为4.4%。

（2）此外，我们的架构设计比所有的初始模型都简单得多，并且需要手动设置的超参数也少得多。

### 5.2. Experiments on ImageNet-5K—在ImgaeNet-5K的实验 

### 翻译 

ImageNet-1K上的性能似乎饱和。但我们认为这不是因为模型的能力，而是因为数据集的复杂性。接下来，我们在具有5000类的更大的想象中子集上评估我们的模型。  
我们的5k数据集是完整的Imagenet-22k集合\[33\]的子集。 5000类别由原始Imagenet1K类别和其他4000类包含完整的ImageNet集中最多的图像。 5K套装有680万个图像，约为1K集的5倍。没有官方火车/ val拆分，因此我们选择评估原始ImageNet-1K验证集。在此1k-class val集合中，可以将模型评估为5k路分类任务（预测为其他4K类的所有标签自动错误）或作为1k路分类任务（SoftMax仅在1K上应用课程）在测试时间。  
实验细节和第4节相同。5K-trainning 模型均从零开始训练并且针对与1K训练模型相同数量的小批量进行训练（因此1/5Xepochs）  
表6和图6显示了在保持复杂性的情况下的比较。ResNeXt-50比ResNet-50减少了3.2%的5K路top-1错误，ResNetXt-101比ResNet-101减少了2.3%的5K路top-1错误。在1K路误差上观察到类似的间隙。这些显示了ResNeXt更强的代表性。  
此外，我们发现在5K集上训练的模型(表6中的1K路误差为22.2%/5.7%)与在1K集上训练的模型(表3中的21.2%/5.6%)相比表现更有竞争力，在验证集上的相同1K路分类任务上进行了评估。这一结果是在不增加训练时间(由于相同数量的小批量)和不进行微调的情况下实现的。我们认为这是一个有希望的结果，因为将5K分类的训练任务是一个更具挑战性的任务。

### 精读 

原因：在ImageNet-1K数据集上的结果似乎饱和了，但是我们认为这不是因为模型的能力不够，而是因为数据集的复杂度不够。下面我们就在更大的ImageNet-5K上进行实验。

方法：在这个验证集上，虽然只有1000类，我们可以当做5000类来分类，如果结果是4000类中的，那么分类错了。

实验结果：

![](https://i-blog.csdnimg.cn/blog_migrate/6a006952a7c6cb761e304a8e3b1dbab7.png)

结论：

（1）在5K数据集上训练的模型与在1K数据集上训练的模型效果相当

（2）这表明分类5k类是更具挑战性的任务。

### 5.3. Experiments on CIFAR—在CIFAR的实验 

### 翻译 

我们在CIFAR-10和100数据集上进行了更多的实验\[23\]。我们使用\[14\]中的体系结构，并用bottleneck模板替换基本残差块  
Bottleneck template： 

![](https://i-blog.csdnimg.cn/blog_migrate/92974b29ff7984d0713695c706837499.png)

我们的网络从单个3×3 conv层开始，然后是3个阶段，每个阶段有3个残差块，最后是平均池和一个完全连接的分类器(总共29层)，如下文\[14\]。我们将这种转换和翻转数据增强称为\[14\]。实施细节见附录。  
我们基于上述基线比较了两种情况下提高复杂性：（i）增加基数并修复所有宽度，或（ii）增加bottleneck的宽度和修复基数= 1.我们在这些变化下培训并评估一系列网络。图7显示了测试误差速率与模型大小的比较。我们发现，随着我们在Imagenet-1K上观察到的宽度，增加的基数比增加的宽度更有效。表7显示了结果和模型尺寸，与宽Reset \[43\]相比，这是最好的已发布记录。我们具有类似型号（34.4M）的模型显示结果比Reset更好。我们的较大方法在CIFAR-10和CIFAR-100上实现了3.58％的测试误差（平均10次运行）和17.31％。据我们所知，这些是文献中的最先进的结果（具有类似的数据增强），包括未发表的技术报告。

### 精读 

数据集：CIFAR-10和CIFAR-100

方法：

增加基数，固定所有的宽度

增加宽度，固定基数为1。

实验结果：

![](https://i-blog.csdnimg.cn/blog_migrate/bab6c6e8fbc8933acdd4775b5804e32c.png)

结论：增加基数是一个更高效的办法

### 5.4. Experiments on COCO object detection—在COCO目标检测的实验 

### 翻译 

接下来，我们评估Coco对象检测集的概括性\[27\]。我们在80K训练集上培训模型加上35k Val子集并在5K Val子集（称为MINIVAL）上进行评估，如\[1\]。我们评估Cocostyle平均精度（AP）以及AP@iou=0.5 \[27\]。我们采用基本更快的R-CNN \[32\]并关注\[14\]将Reset / Resnext插入其中。模型在ImageNet-1k上预先培训，并在检测集上进行微调。实施细节在附录中。  
表8显示了比较。在50层基线上，RESNEXT将AP @ 0.5改善2.1％，AP达1.0％，而不会增加复杂性。 ResNext在101层基线上显示了较小的改进。我们猜测更多培训数据将导致更大的差距，如想象的-5K集合所观察到的。

### 精读 

操作：

在80k训练集+35kval训练，在5k val上测试。

使用了标准的Faster R-CNN，直接将ResNet/ResNeXt替换backbone。

模型在ImageNet上预训练过，在检测数据集上微调。

结果：

![](https://i-blog.csdnimg.cn/blog_migrate/1911d534a467633081ef8caef1afa9a8.png)

结论：使用更多的训练数据，会增加提升效果。

## 论文十问 

> Q1：论文试图解决什么问题？
> 
> 本文用一种平行堆叠相同拓扑结构的blocks代替原来 ResNet 的三层卷积的block，在不明显增加参数量级的情况下提升了模型的准确率，同时由于拓扑结构相同，超参数也减少了，便于模型移植。

> Q2：这是否是一个新的问题？
> 
> 不是，是对ResNet的一个改进，结合了Inception和ResNet优点

> Q3：这篇文章要验证一个什么科学假设？
> 
> 1.在保持相同的计算复杂度和模型尺寸条件下，我们提出的这个网络模块要比原来的 ResNet 模块性能要好。
> 
> 2.增加基数是一种比更深或更宽更有效的获得准确性的方法

> Q4：有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？
> 
> VGG/ResNet/Inception。图像分类

> Q5：论文中提到的解决方案之关键是什么？
> 
> （1）提炼VGG、ResNet和Inception系列的优秀思想；
> 
> （2）处理相同尺寸的特征图时，采用同样大小、数量的卷积核；
> 
> （3）特征图分辨率长宽降低2倍时，特征图通道数（卷积核数量）翻倍；
> 
> （4）提出cardinality来衡量模型复杂度，实验表明cardinality比模型深度、宽度更高效

> Q6：论文中的实验是如何设计的？
> 
> 以ResNet50为基准，在ImageNet-1K、ImgaeNet-5K、CIFAR、COCO这些数据集上做对比实验。

> Q7：用于定量评估的数据集是什么？代码有没有开源？
> 
> ImageNet-1K、ImgaeNet-5K、CIFAR、COCO。有开源

> Q8：论文中的实验及结果有没有很好地支持需要验证的科学假设？
> 
> 验证了。实验结果和ResNet对比有提升，参数也减少了。

> Q9：这篇论文到底有什么贡献？
> 
> 采用分支同构使网络结构简明，模块化需要手动调节的超参数少；提炼split-transform-merge思想；引入cardinality指标，为CNN模型提供新的思路。

> Q10：下一步呢？有什么工作可以继续深入？
> 
> 最近ResNeXt已经应用到Mask R-CNN中，取得了SOTA的COCO实例分割和目标检测效果。

代码复现：[ResNeXt代码复现＋超详细注释（PyTorch）][ResNeXt_PyTorch]


[VGGNet]: https://blog.csdn.net/weixin_43334693/article/details/128148803?spm=1001.2014.3001.5501
[GoogLeNet InceptionV1]: https://blog.csdn.net/weixin_43334693/article/details/128267380?spm=1001.2014.3001.5501
[ResNet]: https://blog.csdn.net/weixin_43334693/article/details/128401720?spm=1001.2014.3001.5501
[Aggregated Residual Transformations for Deep Neural Networks _arxiv.org]: https://arxiv.org/abs/1611.05431
[GitHub - facebookresearch_ResNeXt_ Implementation of a classification framework from the paper Aggregated Residual Transformations for Deep Neural Networks]: https://github.com/facebookresearch/ResNeXt
[Link 1]: #%E5%89%8D%E8%A8%80
[Abstract]: #Abstract%E2%80%94%E6%91%98%E8%A6%81
[Introduction]: #%E4%B8%80%E3%80%81Introduction%E2%80%94%E7%AE%80%E4%BB%8B
[Related Work]: #%E4%BA%8C%E3%80%81Related%20Work%E2%80%94%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C
[2.1Multi-branch convolutional networks]: #2.1Multi-branch%20convolutional%20networks%E2%80%94%E5%A4%9A%E5%88%86%E6%94%AF%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C
[2.2Grouped convolutions]: #2.2Grouped%20convolutions%E2%80%94%E5%88%86%E7%BB%84%E5%8D%B7%E7%A7%AF
[2.3Compressing convolutional networks]: #2.3Compressing%20convolutional%20networks%E2%80%94%E5%8E%8B%E7%BC%A9%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C
[2.4Ensembling]: #2.4Ensembling%E2%80%94%E9%9B%86%E6%88%90
[Method]: #%E4%B8%89%E3%80%81Method%E2%80%94%E6%96%B9%E6%B3%95
[3.1. Template]: #3.1.%20Template%E2%80%94%E6%A8%A1%E5%9E%8B
[3.2. Revisiting Simple Neurons]: #3.2.%20Revisiting%20Simple%20Neurons%E2%80%94%E5%9B%9E%E9%A1%BE%E5%8D%95%E4%B8%AA%E7%A5%9E%E7%BB%8F%E5%85%83
[3.3. Aggregated Transformations]: #3.3.%20Aggregated%20Transformations%E2%80%94%E8%81%9A%E5%90%88%E5%8F%98%E6%8D%A2
[3.4. Model Capacity]: #3.4.%20Model%20Capacity%E2%80%94%E6%A8%A1%E5%9E%8B%E8%83%BD%E5%8A%9B
[Implementation details]: #%E5%9B%9B%E3%80%81Implementation%20details%E2%80%94%E6%89%A7%E8%A1%8C%E7%BB%86%E8%8A%82
[Experiments]: #%E4%BA%94%E3%80%81Experiments%E2%80%94%E5%AE%9E%E9%AA%8C
[5.1. Experiments on ImageNet-1K_ImageNet-1K]: #5.1.%20Experiments%20on%20ImageNet-1K%E2%80%94%E5%9C%A8ImageNet-1K%E7%9A%84%E5%AE%9E%E9%AA%8C
[5.2. Experiments on ImageNet-5K_ImgaeNet-5K]: #5.2.%20Experiments%20on%20ImageNet-5K%E2%80%94%E5%9C%A8ImgaeNet-5K%E7%9A%84%E5%AE%9E%E9%AA%8C
[5.3. Experiments on CIFAR_CIFAR]: #5.3.%20Experiments%20on%20CIFAR%E2%80%94%E5%9C%A8CIFAR%E7%9A%84%E5%AE%9E%E9%AA%8C
[5.4. Experiments on COCO object detection_COCO]: #5.4.%20Experiments%20on%20COCO%20object%20detection%E2%80%94%E5%9C%A8COCO%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%9A%84%E5%AE%9E%E9%AA%8C
[Link 2]: #%E8%AE%BA%E6%96%87%E5%8D%81%E9%97%AE
[ResNeXt_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128664382?spm=1001.2014.3001.5501