## ![](https://i-blog.csdnimg.cn/blog_migrate/9efdfa7ce0c67fc76b0a2d7f739eae7e.jpeg) 

## 前言 

在CNN网络结构的演化上，出现过许多优秀的CNN网络，CNN的经典结构始于1998年的LeNet，成于2012年历史性的AlexNet，从此盛于图像相关领域。

发展历史：Lenet --> Alexnet --> ZFnet --> VGG --> NIN --> GoogLeNet -->ResNet--> DenseNet -->ResNeXt ---> EfficientNet

<table> 
 <thead> 
  <tr> 
   <th>神经网络</th> 
   <th>年份</th> 
   <th>标签</th> 
   <th>作者</th> 
  </tr> 
 </thead> 
 <tbody> 
  <tr> 
   <td><strong>LeNets</strong></td> 
   <td>1998年</td> 
   <td>CNN开山之作</td> 
   <td>纽约大学</td> 
  </tr> 
  <tr> 
   <td><strong>AlexNet</strong></td> 
   <td>2012年</td> 
   <td> <p>深度学习CV领域划时代论文</p> <p>具有里程碑意义</p> <p>ImageNet 2020冠军</p> </td> 
   <td>多伦多大学　Hinton团队</td> 
  </tr> 
  <tr> 
   <td><strong>ZFNet</strong></td> 
   <td>2013年</td> 
   <td>ImageNet 2013冠军</td> 
   <td>纽约大学</td> 
  </tr> 
  <tr> 
   <td><strong>GoogLeNet V1</strong></td> 
   <td>2014年</td> 
   <td> <p>Google系列论文开创论文</p> <p>ImageNet 2014冠军　</p> <p>Inception模块</p> </td> 
   <td>谷歌</td> 
  </tr> 
  <tr> 
   <td><strong>VGG</strong></td> 
   <td>2014年</td> 
   <td> <p>开启3*3卷积堆叠时代</p> <p>ImageNet 2014亚军　</p> <p>VGG-16和VGG-19</p> </td> 
   <td>牛津大学</td> 
  </tr> 
  <tr> 
   <td><strong>NiN</strong></td> 
   <td>2014年</td> 
   <td> <p>Network in Network（网中网）　</p> <p>GAP（全局平均池化）</p> </td> 
   <td>新加坡国立大学</td> 
  </tr> 
  <tr> 
   <td><strong>ResNet</strong></td> 
   <td>2015年</td> 
   <td> <p>最具影响力的卷积神经网络</p> <p>ImageNet 2015冠军　</p> <p>残差网络</p> </td> 
   <td>何凯明团队　微软亚院</td> 
  </tr> 
  <tr> 
   <td><strong>Wide ResNet</strong></td> 
   <td>2016年</td> 
   <td>增加残差中卷积核数量（宽度）</td> 
   <td>谢尔盖·扎戈鲁科</td> 
  </tr> 
  <tr> 
   <td><strong>Squeeze Net</strong></td> 
   <td>2016年</td> 
   <td> <p>轻量级网络</p> <p>压缩参数量</p> </td> 
   <td>斯坦福大学　伯克利大学</td> 
  </tr> 
  <tr> 
   <td><strong>DenseNet</strong></td> 
   <td>2017年</td> 
   <td> <p>ImageNet 2016冠军</p> <p>CVPR 2017最佳论文　</p> <p>Dense模块</p> </td> 
   <td> <p>&nbsp; &nbsp; &nbsp; &nbsp;康奈尔大学　清华大学</p> <p></p> </td> 
  </tr> 
  <tr> 
   <td><strong>SENet</strong></td> 
   <td>2017年</td> 
   <td> <p>ImageNet 2017冠军　</p> <p>SE模块</p> </td> 
   <td>momenta + 牛津大学　胡杰</td> 
  </tr> 
  <tr> 
   <td><strong>ResNext</strong></td> 
   <td>2017年</td> 
   <td> <p>何恺明团队对ResNet重大改进</p> <p>Resnet + Inception</p> </td> 
   <td>Saining Xie团队</td> 
  </tr> 
  <tr> 
   <td><strong>FractalNet</strong></td> 
   <td>2017年</td> 
   <td>分形网络</td> 
   <td>Gustav Larsson团队</td> 
  </tr> 
  <tr> 
   <td><strong>MobileNets</strong></td> 
   <td>2017年</td> 
   <td> <p>轻量级　</p> <p>Group卷积</p> <p>Depthwise Seperable卷积</p> </td> 
   <td>谷歌</td> 
  </tr> 
  <tr> 
   <td><strong>NASNet</strong></td> 
   <td>2018年</td> 
   <td>神经架构搜索　强化学习</td> 
   <td>谷歌</td> 
  </tr> 
 </tbody> 
</table>

![](https://i-blog.csdnimg.cn/blog_migrate/d47db05a199ce9e023fa694b691b41bd.jpeg)（图片来源见水印）

在前一段的学习中，我们精读了八篇经典网络的论文，并实现了代码复现。这篇文章是对之前学习的一个总结，主要针对各个网络结构和创新点以及如何实现来进行一个回顾。

## 目录 

[前言][Link 1]

[一、AlexNet][AlexNet]

[1.简介 ][1._]

[2.网络结构][2.]

[3.创新点][3.]

[ 二、VGG][_VGG]

[1.简介][1.]

[2.网络结构 ][2._]

[3.网络配置][3. 1]

[ 4.创新点][4.]

[ 三、GoogLeNet InceptionV1][_GoogLeNet InceptionV1]

[1.简介][1.]

[2.网络结构 ][2._]

[3.网络配置][3. 2]

[4.创新点 ][4._]

[ 四、GoogLeNet InceptionV2-V3][_GoogLeNet InceptionV2-V3]

[1.简介][1.]

[ 2.网络结构 ][2._ 1]

[3.网络配置][3. 1]

[ 4.创新点 ][4._ 1]

[ 五、ResNet][_ResNet]

[1.简介][1.]

[ 2.网络结构 ][2._ 1]

[3.网络配置][3. 1]

[ 4.创新点 ][4._ 1]

[ 六、DenseNet][_DenseNet]

[1.简介][1.]

[ 2.网络结构 ][2._ 1]

[3.网络配置 ][3._]

[ 4.创新点 ][4._ 1]

[ 七、SENet][_SENet]

[1.简介][1.]

[2.网络结构 ][2._]

[3.网络配置 ][3._]

[ 4.创新点 ][4._ 1]

[ 八、ResNeXt][_ResNeXt]

[1.简介][1.]

[2.网络结构 ][2._]

[3.网络配置 ][3._]

[4.创新点 ][4._ 1]

## 🚀一、AlexNet 

期刊日期：NIPS-2012

论文名称：《ImageNet Classification with Deep Convolutional Neural Networks》

论文精读： [经典神经网络论文超详细解读（一）——AlexNet学习笔记（翻译＋精读）][AlexNet 1]

### 1.简介  

2012年Geoffrey和他学生Alex在ImageNet的竞赛中，刷新了image classification的记录，一举夺得冠军，自此打开了深度学习的大门，也使得深度学习成为学术界的新宠。这次竞赛中Alex所用的结构就被称为作为AlexNet。

### 2.网络结构 

![](https://i-blog.csdnimg.cn/blog_migrate/7d2aaccf71b4de5ba67e9b967ecd17d6.png)

第一层卷积层使用96个大小为11x11x3的卷积核对224x224x3的输入图像以4个像素为步长（这是核特征图中相邻神经元感受域中心之间的距离）进行滤波。

第二层卷积层将第一层卷积层的输出（经过响应归一化和池化）作为输入，并使用256个大小为5x5x48的核对它进行滤波。

第三层、第四层和第五层的卷积层在没有任何池化或者归一化层介于其中的情况下相互连接。第三层卷积层有384个大小为3x3x256的核与第二层卷积层的输出（已归一化和池化）相连。第四层卷积层有384个大小为3x3x192的核，第五层卷积层有256个大小为 的核。每个全连接层有4096个神经元。

### 3.创新点 

（1）使用非线性激活函数ReLU加快了训练速度：AlexNet使用ReLU代替了Sigmoid,其能更快的训练,同时解决sigmoid在训练较深的网络中出现的梯度消失。

（2）使用多个GPU并行训练：加速了计算。

（3）LRN局部归一化：增加了泛化能力，做了平滑处理，提高了1%~2%的识别率, LRN 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大（使得重要位置更突出），并抑制其他反馈较小的神经元。

（4）重叠池化：以前的CNN中普遍使用平均池化层，AlexNet全部使用最大池化层,避免了平均池化层的模糊化的效果,并且步长比池化的核的尺寸小,这样池化层的输出之间有重叠,提升了特征的丰富性。

（5）Dropout：避免过拟合，Dropout随机失活，随机忽略一些神经元,以避免过拟合。

（6）数据增广：来增加模型泛化能力 256×256×3 -->随机裁剪224×224×3 -->进入网络。

## 🚀二、VGG 

期刊日期：ICLR-2015

论文名称：《Very Deep Convolutional Networks for Large-Scale Image Recognition》

论文精读：[经典神经网络论文超详细解读（二）——VGGNet学习笔记（翻译＋精读）][VGGNet]

### 1.简介 

VGG-Net来自 Andrew Zisserman 教授的组 (Oxford)，在AlexNet基础上提出了更深的网络，分别为VGG-16和VGG-19。在2014年的 ILSVRC 图像定位和分类两个问题上分别取得了第一名和第二名。相比于AlexNet，AlexNet只有8层，而VGG有16～19层；AlexNet使用了11x11的卷积核，VGG使用了3x3卷积核和2x2的最大池化层，VGG参数是AlexNet的三倍。为后面的框架提供了方向：加深网络的深度。

### 2.网络结构  

![](https://i-blog.csdnimg.cn/blog_migrate/9fa2f26fb6739b549bf6ef133f240697.png)

以VGG16为例：

1.输入：一个固定大小的224\*224RGB图像

2.唯一预处理：将输入的224×224×3通道的像素值，减去平均RGB值，然后进行训练

3.卷积核：①使用最小尺寸的卷积核3×3这个尺寸捕捉上下左右和中间方位的最小尺寸

②有些卷积层中还使用了1×1大小的卷积核（FC层之间的），可看作是输入通道的线性变换

4.卷积步幅 ：卷积步幅固定为1像素，3×3卷积层的填充设置为1个像素。

5.池化层：池化层采用空间池化，空间池化有五个最大池化层，他们跟在一些卷积层之后，但是也不是所有的卷积层后都跟最大池化。最大池化层使用2×2像素，步幅为2。

6.卷积层：一个卷积层之后是三个全连接层：前两个层各有4096个通道，第三个层执行1000路ILSVRC分类，因此包含1000个通道（每类一个）（1000分类对应ImageNet-1K）。最后一层是softmax层，用来分类。在所有网络中，完全连接的层的配置都是相同的。

7.隐藏层：所有隐藏层都有ReLU非线性函数，网络除了第一个其他都不包含局部响应归一化（LRN）

### 3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/7d1f53a6b5bf11ca4e2d47e337304c0f.png)

###  4.创新点 

1.用三个3×3的卷积核代替7×7的卷积核，有的FC层还用到了1×1的卷积核以及2×2的池化层。网络更深，增加了CNN对特征的学习能力。

2.在更深的结构中没有用到LRN（推翻了Alex），避免了部分内存和计算的增加。

3.VGG采用的是一种Pre-training的方式，先训练级别简单（层数较浅）的VGGNet的A级网络，然后使用A网络的权重来初始化后面的复杂模型，加快训练的收敛速度。

4.采用了Multi-Scale的方法来训练和预测。可以增加训练的数据量，防止模型过拟合，提升预测准确率 。

## 🚀三、GoogLeNet InceptionV1 

期刊日期：CVPR-2015

论文名称：《Going deeper with convolutions》

论文精读：[经典神经网络论文超详细解读（三）——GoogLeNet InceptionV1学习笔记（翻译＋精读+代码复现）][GoogLeNet InceptionV1]

代码复现：[GoogLeNet InceptionV1代码复现＋超详细注释（PyTorch）][GoogLeNet InceptionV1_PyTorch]

### 1.简介 

GoogLeNet在2014的ImageNet分类任务上击败了VGG夺得冠军，GoogLeNet跟AlexNet,VGG-Nets这种单纯依靠加深网络结构进而改进网络性能的思路不一样，它在加深网络的同时（22层），在同一层中使用了多个不同尺寸的卷积，以获得不同的视野，最后级联（直接叠加通道数量）。引入Inception结构代替了单纯的卷积+激活的传统操作（这思路最早由Network in Network提出）

### 2.网络结构  

![](https://i-blog.csdnimg.cn/blog_migrate/535bd2c3c4a45688f04dc7c993803d6b.png)

1.输入：特征图先被复制成4份并分别被传至接下来的4个部分,将前一层用三个单独的卷积核扫描，并加一个池化操作，然后把这四个操作的输出串联到一起作为下一层的输入。

2.卷积层：之所以卷积核大小采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定padding=0、1、2，那么卷积之后便可以得到相同维度的特征，方便最后拼接在一起（最后拼接在一起，也就是concat操作，需要让四条支路的特征的维度相同，这样才能融合）。

3.池化层：文章说很多地方都表明pooling挺有效，所以Inception里面也嵌入了。

4.输出：GoogLeNet采用的是全局平均池化层，得到的是高和宽均为1的卷积层，然后添加丢弃概率为40%的Dropout，输出层激活函数采用的是softmax。

### 3.网络配置   ![](https://i-blog.csdnimg.cn/blog_migrate/c32cdc61f7b9a9d2f96b244e22d2bc7b.png) 

1.第一模块采用的是1个单纯的卷积层＋1个最大池化层。

2.第二模块采用2个卷积层＋1个最大池化层。

3.第三模块Inception 3a/3b层，分为四个分支，采用不同尺度。

4.第四模块Inception 4a/4b/4c/4e，与Inception3a/3b类似。

5.第五模块Inception 5a/5b，与Inception3a/3b类似。

6.辅助分类器：为了利用中间层抽象的特征，在某些中间层中添加含有多层的分类器。红色边框内部代表添加的辅助分类器。GoogLeNet中共增加了两个辅助的softmax分支。

### 4.创新点  

（1）引入了Inception，使用更宽更深的网络，提升网络性能；

（2）感受野的大小不同，获得不同尺度特征；

（3）用稀疏连接代替密集连接，减少计算资源需求；

（4）1\*1的卷积核降维，提高计算资源利用率；

（5）添加两个辅助分类器帮助训练，避免梯度消失；

（6）后面的全连接层全部替换为简单的全局平均pooling。

## 🚀四、GoogLeNet InceptionV2-V3 

期刊日期：CVPR-2016

论文名称：《Rethinking the Inception Architecture for Computer Vision》

论文精读：[经典神经网络论文超详细解读（四）——InceptionV2-V3学习笔记（翻译＋精读＋代码复现）][InceptionV2-V3]

代码复现：[ GoogLeNet InceptionV3代码复现＋超详细注释（PyTorch）][GoogLeNet InceptionV3_PyTorch]

### 1.简介 

GoogLeNet经过了Inception V1、Inception V2（BN）的发展以后，Google的Szegedy等人又对其进行了更深层次的研究和拓展，在本文中，作者提出了当前环境下，网络设计的一些重要准则，并根据这些准则，对原有的GoogLeNet进行了改进，提出了一个更加复杂、性能更好的模型框架：Inception V3。这篇文章证明了这些改进的有效性，并为以后的网络设计提供了新的思路。

### 2.网络结构  

### ![](https://i-blog.csdnimg.cn/blog_migrate/015e6f78f72b89f307f1056e85894c99.png)![](https://i-blog.csdnimg.cn/blog_migrate/9fbd25b6fddbe1cdaaa5b3abe006d942.png) 

左图：分解成更小的卷积。将一个5×5的卷积换成两个3×3卷积，同理7×7卷积可以用3个3×3卷积。

右图：非对称分解卷积。 将3x3卷积可分解为1x3和3x1两个不对称卷积（空间可分离卷积）。

###    3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/8c467b75d9dcd3b5f146add9df2e2886.png)

Inception-v2：相比起Inception-v1结构，这里将开始的7x7卷积使用3个3x3卷积替代，在后面的Inception模块中分别使用了三种不同的模式：

第一部分输入的特征图尺寸为35x35x288,采用了图5中的架构，将5x5以两个3x3代替。

第二部分输入特征图尺寸为17x17x768，采用了图6中nx1+1xn的结构

第三部分输入特征图尺寸为8x8x1280, 采用了图7中所示的并行模块的结构

![](https://i-blog.csdnimg.cn/blog_migrate/24e80543a29d0248127acfc32c70156e.png)

Inception-v3：对Inception-v2进行改进  
（1）InceptionV2 加入RMSProp(一种计算梯度的方法)

（2）在上面的基础上加入Label Smoothing(LSR,标签平滑正则化)

（3）在上面的基础上再加入7×7的卷积核分解(分解成3×3)

（4）在上面的基础上再加入含有BN的辅助分类器

所以本文最终提出的InceptionV3=inceptionV2+RMSProp优化+LSR+BN-auxilary

### 4.创新点  

（1）卷积核分解：1.将大的卷积核分解成小的。2.非对称分解（n\*n的卷积核替换成 1\*n 和 n\*1 的卷积核堆叠）

（2）并行执行（卷积C+池化P），再进行feature map的堆叠

（3）标签平滑（LSR）进行模型正则化

（4）纠正了辅助分类器的作用，靠近输入辅助分类器（加不加没有影响），但是靠近输出的辅助分类器（加上BN后）可以起到正则化的作用。

## 🚀五、ResNet 

期刊日期：CVPR-2016

论文名称：《Deep Residual Learning for Image Recognition》

论文精读：[经典神经网络论文超详细解读（五）——ResNet（残差网络）学习笔记（翻译＋精读＋代码复现）][ResNet]

代码复现：[ResNet代码复现＋超详细注释（PyTorch）][ResNet_PyTorch]

### 1.简介 

2015年何恺明等大神推出的ResNet相当经典，在深度学习发展历程中具有里程碑的意义。在ISLVRC和COCO上横扫所有选手，获得冠军，同时也在2016CVPR获得best paper。ResNet在网络结构上做了大创新，首次提出了残差的思想（跨层连接，即![H(x)=F(x)+x](https://i-blog.csdnimg.cn/blog_migrate/31505f86b887582b80008f76f95dca70.gif)），不再是简单的堆积层数，解决了网络过深而导致的梯度消失的问题，为更深的网络提供了有力的方向。

### 2.网络结构  

###  

![](https://i-blog.csdnimg.cn/blog_migrate/2d0b0f311405382eff0e175e928257ba.png)

左边的残差结构适用于网络层数较浅的网络，也就是ResNet-34，右边的适用于ResNet-50/101/152。这个是进行一个相加操作，不是拼接操作

残差模块：一条路不变（恒等映射）；另一条路负责拟合相对于原始网络的残差，去纠正原始网络的偏差，而不是让整体网络去拟合全部的底层映射，这样网络只需要纠正偏差。

本质：在ResNet中，传递给下一层的输入变为H(x)=F(x)+x，即拟合残差F(x)=H(x)－x  
（1）加了残差结构后，给了输入x一个多的选择。若神经网络学习到这层的参数是冗余的时候，它可以选择直接走这条“跳接”曲线（shortcut connection），跳过这个冗余层，而不需要再去拟合参数使得H(x)=F(x)=x

（2）加了恒等映射后，深层网络至少不会比浅层网络更差。

（3）而在Resnet中，只需要把F(x)变为0即可，输出变为F(x)+x=0+x=x很明显，将网络的输出优化为0比将其做一个恒等变换要容易得多。

### 3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/37ae477211d5e3d45a6f4078d4039c98.png)

Plain网络：plain网络结构主要受VGG网络的启发。 卷积层主要为3\*3的卷积核，直接通过stride为2的卷积层来进行下采样。在网络的最后是一个全局的平均pooling层和一个1000类的包含softmax的全连接层。加权层的层数为34。

残差网络：在plain网络的基础上，加入shortcuts连接，就变成了相应的残差网络如上图，实线代表维度一样，直接相加 。虚线代表维度不一样（出现了下采样，步长为2的卷积），使用残差网络

### 4.创新点  

（1）实现了超深的网络结构（突破1000层）

（2）提出了残差网络，将x直接传递到后面的层，使得网络可以很容易的学习恒等变换，从而解决网络退化的问题，同时也使得学习效率更高。

（3）使用BatchNormalization加速训练（丢弃dropout）

## 🚀六、DenseNet 

期刊日期：CVPR-2017

论文名称：《Densely Connected Convolutional Networks》

论文精读：[经典神经网络论文超详细解读（六）——DenseNet学习笔记（翻译＋精读＋代码复现）][DenseNet]

代码复现：[DenseNet代码复现＋超详细注释（PyTorch）][DenseNet_PyTorch]

### 1.简介 

DenseNet是CVPR 2017的best paper，主要还是和ResNet及Inception网络做对比，比ResNet来的更加彻底，即当前的每一层都和前面的每一层连接，在CIFAR指标上全面超越ResNet。可以说DenseNet吸收了ResNet最精华的部分，并在此上做了更加创新的工作，使得网络性能进一步提升。

### 2.网络结构  

### ![](https://i-blog.csdnimg.cn/blog_migrate/1f197a980e5bd63b1fcbcce4d06733a6.png) 

DenseNet的网络结构主要由Dense Block＋Transition组成

DenseBlock（定义了输入输出如何连接） 是包含很多层的模块，每个层的特征图大小相同，层与层之间采用稠密连接方式。

（1）DenseBlock：BN+ReLU+一个3×3的Conv

（2）Bottleneck层：BN＋Relu＋1×1conv＋BN＋Relu＋3×3conv

Transition模块（控制通道数） 是连接两个相邻的Dense Block，并且通过Pooling使特征图大小降低。  
转换层=BN+1×1Conv+2×2AveragePooling

将Dense Block+bottleneck+Translation的模型称为DenseNet-BC

### 3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/617dafa60b353f13d9bd80d0ed409e93.png)

DenseNet共包含三个Dense Block，各个模块的特征图大小分别为 32 × 32，16×16和8×8，每个Dense Block里面的层数相同（Dense Block本身不会改变特征图尺寸，所以是转换层导致的尺寸变化）

1.在进入第一个Dense Block时，对输入图像进行16（或为DenseNet-BC增长率的两倍）输出通道的卷积  
2.对于3x3卷积，使用zero-padding，以保证特征图的大小不变  
3.使用1x1卷积＋2x2平均池化作为两个Dense Block的转换层  
4.在最后一个Dense Block的末端，执行一个全局平均池化＋一个softmax分类器。

### 4.创新点  

（1）前馈方式连接：它建立的是前面所有层与后面层的密集连接（dense connection）即在这个模块中才进行每一层的连接，这样便于控制输入尺寸的大小，Dense block模块之间就可以直接使用池化操作了

（2）特征重用（feature reuse）：引入了具有相同特征图大小的任意两层之间的直接连接，产生易于训练和高效的参数压缩模型。将不同层学习的feature-map串联起来，增加了后续层输入的变化，提高了效率。

## 🚀 七、SENet 

期刊日期：TPAMI-2017

论文名称：《Squeeze-and-Excitation Networks》

论文精读：[ 经典神经网络论文超详细解读（七）——SENet（注意力机制）学习笔记（翻译＋精读＋代码复现）][_SENet 1]

代码复现：[SENet代码复现＋超详细注释（PyTorch）][SENet_PyTorch]

### 1.简介 

SENet block引入了注意力机制，它的出现使卷积神经网络又一次取得较大进步，但它并不是一个完整的网络结构，而是一个即插即用的小插件，可以嵌到其他分类或检测模型中，作者采用SENet block和ResNeXt结合在ILSVRC 2017的分类项目中拿到第一，在ImageNet数据集上将top-5 error降低到2.251%，原先的最好成绩是2.991%。

### 2.网络结构  

### ![](https://i-blog.csdnimg.cn/blog_migrate/8e8f9ebb379cc67064da270eb1105b1a.png) 

1.Ftr—传统卷积操作：对于一个C×W×H的输入X，在经过Ftr卷积操作之后，得到的输出Uc（也就是C个大小为H×W的特征图）

2.Fsq—压缩：Fsq操作就是使用通道的全局平均池化，将包含全局信息的W×H×C 的特征图直接压缩成一个1×1×C的特征向量，即将每个二维通道变成一个具有全局感受野的数值，此时1个像素表示1个通道，屏蔽掉空间上的分布信息，更好的利用通道间的相关性。

3.Fex—激励：采用两个全连接层+两个激活函数组成的结构输出和输入特征同样数目的权重值，也就是每个特征通道的权重系数。(1)第一个FC层：ReLU （δ）(2)第二个FC层：Sigmoid（σ）

4.Fscale —权重：将Excitation输出的权重看做每个特征通道的重要性，也就是对于U每个位置上的所有H×W上的值都乘上对应通道的权值，完成对原始特征的重校准。

### 3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/4fa84fcf03896ec0b1b8c84bf02f25b5.png)

1.首先由 Inception结构 或 ResNet结构处理后的C×W×H特征图开始，通过Squeeze操作对特征图进行全局平均池化（GAP），得到1×1×C 的特征向量

2.紧接着两个 FC 层组成一个 Bottleneck 结构去建模通道间的相关性：

经过第一个FC层，将C个通道变成 C/ r ，减少参数量，然后通过ReLU的非线性激活，到达第二个FC层

经过第二个FC层，再将特征通道数恢复到C个，得到带有注意力机制的权重参数

3.最后经过Sigmoid激活函数，最后通过一个 Scale 的操作来将归一化后的权重加权到每个通道的特征上。

### 4.创新点  

（1）研究了网络设计的一个不同方面——通道之间的关系

（2）引入了一种叫做SE block的新的结构单元，它能够建模特征图通道间的相互依赖关系，其目标是通过显式建模卷积特征通道之间的相互依赖来提高网络产生的表征的质量。

（3）提出了注意力机制，允许网络进行特征重新校准，通过该机制网络可以学习使用全局信息，选择性地强调信息特征和抑制无用的特征

## 🚀 八、ResNeXt 

期刊日期：CVPR-2017

论文名称：《Aggregated Residual Transformations for Deep Neural Networks》

论文精读：[经典神经网络论文超详细解读（八）——ResNeXt学习笔记（翻译＋精读＋代码复现）][ResNeXt]

代码复现：[ResNeXt代码复现＋超详细注释（PyTorch）][ResNeXt_PyTorch]

### 1.简介 

ResNeXt是何恺明大神的又一经典之作。提出了一种用于图像分类的简单、高度模块化的网络架构。 这个网络可以被解释为 VGG、ResNet 和 Inception 的结合体，它通过重复多个block（如在 VGG 中）块组成，每个block块聚合了多种转换（如 Inception），同时考虑到跨层连接（来自 ResNet）。该网络通过反复堆叠Building Block实现，Building Block则通过聚集简洁的卷积模块来实现。并且提出一个与网络宽度和深度类似作用的参数，用来衡量网络大小，称之为Cardinality基数。在ILSVRC 2016分类任务获得了第二名。

### 2.网络结构  

### ![](https://i-blog.csdnimg.cn/blog_migrate/be1a362110e992cc16ef37066440b76c.png) 

（a）表示先划分，单独卷积并计算输出，最后输出相加。split-transform-merge三阶段形式

（b）表示先划分，单独卷积，然后拼接再计算输出。将各分支的最后一个1×1卷积聚合成一个卷积。

（c）就是分组卷积。将各分支的第一个1×1卷积融合成一个卷积，3×3卷积采用group（分组）卷积的形式，分组数=cardinality（基数）

（c）结构分析：

（1）首先通过一个1x1的卷积层进行降维处理，将它的channel从256降低到128,

（2）然后在通过group卷积进行处理，这里group 卷积的卷积核为3x3它的groups数为32，它所输出的channel也是等于128,

（3）接着通过1x1卷积对它进行升维

（4）最后将它的输出与我们的输入进行相加得到最后输出。

### 3.网络配置 

![](https://i-blog.csdnimg.cn/blog_migrate/cfd0a2559ebb647c0d5589fa3092facd.png)

32 指进入网络的第一个ResNeXt基本结构的分组数量C（即基数）为32

4d 表示depth即每一个分组的通道数为4（所以第一个基本结构输入通道数为128）

### 4.创新点  

（1）提炼VGG、ResNet和Inception系列的优秀思想。

（2）分支同构，处理相同尺寸的特征图时，采用同样大小、数量的卷积核，网络结构简明，模块化需要手动调节的超参数少。

（3）提出cardinality来衡量模型复杂度，实验表明cardinality比模型深度、宽度更高效。

关于经典神经网络的学习我们就要告一段落了，下一步会开启目标检测的学习，敬请期待吧~

祝大家小年快乐呀！


[Link 1]: #%E5%89%8D%E8%A8%80
[AlexNet]: #%E4%B8%80%E3%80%81AlexNet
[1._]: #1.%E7%AE%80%E4%BB%8B%C2%A0
[2.]: #2.%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84
[3.]: #3.%E5%88%9B%E6%96%B0%E7%82%B9
[_VGG]: #%C2%A0%E4%BA%8C%E3%80%81VGG
[1.]: #1.%E7%AE%80%E4%BB%8B
[2._]: #2.%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%C2%A0
[3. 1]: #3.%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE
[4.]: #%C2%A04.%E5%88%9B%E6%96%B0%E7%82%B9
[_GoogLeNet InceptionV1]: #%C2%A0%E4%B8%89%E3%80%81GoogLeNet%20InceptionV1
[3. 2]: #3.%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE%E2%80%8B%E7%BC%96%E8%BE%91%E2%80%8B
[4._]: #4.%E5%88%9B%E6%96%B0%E7%82%B9%C2%A0
[_GoogLeNet InceptionV2-V3]: #%C2%A0%E5%9B%9B%E3%80%81GoogLeNet%20InceptionV2-V3
[2._ 1]: #%C2%A02.%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%C2%A0
[4._ 1]: #%C2%A04.%E5%88%9B%E6%96%B0%E7%82%B9%C2%A0
[_ResNet]: #%C2%A0%E4%BA%94%E3%80%81ResNet
[_DenseNet]: #%C2%A0%E5%85%AD%E3%80%81DenseNet
[3._]: #3.%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE%C2%A0
[_SENet]: #%C2%A0%E4%B8%83%E3%80%81SENet
[_ResNeXt]: #%C2%A0%E5%85%AB%E3%80%81ResNeXt
[AlexNet 1]: https://blog.csdn.net/weixin_43334693/article/details/128127653?spm=1001.2014.3001.5501
[VGGNet]: https://blog.csdn.net/weixin_43334693/article/details/128148803?spm=1001.2014.3001.5501
[GoogLeNet InceptionV1]: https://blog.csdn.net/weixin_43334693/article/details/128267380?spm=1001.2014.3001.5501
[GoogLeNet InceptionV1_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128293495?spm=1001.2014.3001.5501
[InceptionV2-V3]: https://blog.csdn.net/weixin_43334693/article/details/128333485?spm=1001.2014.3001.5501
[GoogLeNet InceptionV3_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128345872?spm=1001.2014.3001.5501
[ResNet]: https://blog.csdn.net/weixin_43334693/article/details/128401720?spm=1001.2014.3001.5501
[ResNet_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128409032?spm=1001.2014.3001.5501
[DenseNet]: https://blog.csdn.net/weixin_43334693/article/details/128478420?spm=1001.2014.3001.5501
[DenseNet_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128485184?spm=1001.2014.3001.5501
[_SENet 1]: https://blog.csdn.net/weixin_43334693/article/details/128563228?spm=1001.2014.3001.5501
[SENet_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128567913?spm=1001.2014.3001.5501
[ResNeXt]: https://blog.csdn.net/weixin_43334693/article/details/128676690
[ResNeXt_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128664382