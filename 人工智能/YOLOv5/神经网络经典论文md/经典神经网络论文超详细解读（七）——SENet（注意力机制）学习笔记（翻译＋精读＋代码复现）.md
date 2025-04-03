## ![](https://i-blog.csdnimg.cn/blog_migrate/237a86e1d2ea7da8c54d9ae6a9c23afd.jpeg) 

## 前言 

上一期文章中我们介绍了DenseNet，该网络核心在于每一个密集块中的每一层的输入都包含了前面的所有层，这些层通过在通道维度上进行拼接，从而一同作为下一层的输入。这在一定程度上缓解了梯度消失的问题，也由此可以构建更加深层次的神经网络。指路→[经典神经网络论文超详细解读（六）——DenseNet学习笔记（翻译＋精读＋代码复现）][DenseNet]

今天我们继续来学习一种新的网络SENet（《Squeeze-and-Excitation Networks》），它的出现使卷积神经网络又一次取得较大进步，并以极大的优势获得了最后一届ILSVRC（2017）竞赛图像分类任务的冠军。就让我们一起来学习一下吧！

论文地址：[SENet：https://arxiv.org/pdf/1709.01507.pdf][SENet_https_arxiv.org_pdf_1709.01507.pdf]

代码地址：[GitHub - hujie-frank/SENet: Squeeze-and-Excitation Networks][GitHub - hujie-frank_SENet_ Squeeze-and-Excitation Networks]

## 目录 

[前言][Link 1]

[ Abstract—摘要][Abstract]

[一、Introduction—引言][Introduction]

[二、Related Work—相关工作][Related Work]

[Deep architectures—更深架构][Deep architectures]

[Attention and gating mechanisms—注意力和门控机制][Attention and gating mechanisms]

[三、Squeeze-and-Excitation Blocks—挤压和激励块][Squeeze-and-Excitation Blocks]

[Transformation（Ftr）—传统卷积操作][Transformation_Ftr]

[Squeeze: Global Information Embedding（Fsq）—压缩：全局信息嵌入][Squeeze_ Global Information Embedding_Fsq]

[Excitation: Adaptive Recalibration（Fex）—激励：自适应重新校准][Excitation_ Adaptive Recalibration_Fex]

[Scale：Reweight（Fscale ）—权重][Scale_Reweight_Fscale]

[Instantiations—实例][Instantiations]

[四、Model and Computational Complexity—模型和计算复杂度][Model and Computational Complexity]

[计算复杂度][Link 2]

[参数量][Link 3]

[五、Implementation—实施][Implementation]

[六、Experiments—实验][Experiments]

[6.1 Image Classificatio—图像分类][6.1 Image Classificatio]

[6.2 Scene Classification—场景分类][6.2 Scene Classification]

[6.3 Object Detection on COCO—关于COCO的对象检测][6.3 Object Detection on COCO_COCO]

[6.4 Analysis and Interpretation—分析和讨论][6.4 Analysis and Interpretation]

[七、Conclusion—结论][Conclusion]

[论文十问][Link 4]

## Abstract—摘要 

### 翻译 

卷积神经网络是建立在卷积运算操作上，通过融合空间和通道级信息在局部感受野内提取信息特征。为促进网络表征力，最近的一些方法已展示一些增强空间编码的益处。在该工作中，我们聚焦在通道关系上并提出一种新的架构单元，我们称之为“压缩-和-激励”（SE）模块，它通过显示建模通道间的相互依赖自适应地校准通道级特征响应。我们证实通过堆叠这类模块，我们能够构建SENet架构，能够在挑战性的数据集上极好泛化。重要的是，我们发现SE模块对已有的当前最先进的深度架构以极小的额外计算成本极大提升了性能。SENet构成了我们ILSVRC2017年分类任务的提交基础，该分类提交版获得了第一名，并显著将top-5错误率提升至2.251%，完成了相对2016年该类目至少25%的相对提升。代码和模型可在[https：//githorb.com/hujie-frank/senet][https_githorb.com_hujie-frank_senet]找到。

### 精读 

#### 主要内容 

背景： CNN的核心是卷积操作，通过局部感受野的方式来融合空间维度和通道维度的特征，针对空间维度的特征提取方法已经被广泛研究。

本文工作： 本篇论文则把研究重点放在通道维度上，探索通道之间的关系，并提出SE Block（Squeeze-and-Excitation Module），它可以自适应地调整通道维度上的特征。

研究成果： SE Block可嵌入堆叠在很多经典的神经网络中如Inception Module、ResNet等构成SENet，在多个数据集上表现良好，能够在仅需增加少量参数的条件下大幅度提升精度。

比赛结果： 在ILSVRC-2017分类挑战赛上获得冠军，将top-5 error降低至2.251%，相对于2016的冠军下降了大约25%

## 一、Introduction—引言 

### 翻译 

卷积神经网络（CNNs）已被证明是解决各种视觉任务的有效模型\[19,23,29,41\]。对于每个卷积层，沿着输入通道学习一组滤波器来表达局部空间连接模式。换句话说，期望卷积滤波器通过融合空间信息和信道信息进行信息组合，而受限于局部感受野。通过叠加一系列非线性和下采样交织的卷积层，CNN能够捕获具有全局感受野的分层模式作为强大的图像描述。最近的工作已经证明，网络的性能可以通过显式地嵌入学习机制来改善，这种学习机制有助于捕捉空间相关性而不需要额外的监督。Inception架构推广了一种这样的方法\[14,39\]，这表明网络可以通过在其模块中嵌入多尺度处理来取得有竞争力的准确度。最近的工作在寻找更好地模型空间依赖\[1,27\]，结合空间注意力\[17\]。

与这些方法相反，通过引入新的架构单元，我们称之为“Squeeze-and-Excitation” (SE)块，我们研究了架构设计的一个不同方向——通道关系。我们的目标是通过显式地建模卷积特征通道之间的相互依赖性来提高网络的表示能力。为了达到这个目的，我们提出了一种机制，使网络能够执行特征重新校准，通过这种机制可以学习使用全局信息来选择性地强调信息特征并抑制不太有用的特征。

SE构建块的基本结构如图1所示。对于任何给定的变换![F_{tr}:X\rightarrow U](https://latex.csdn.net/eq?F_%7Btr%7D%3AX%5Crightarrow%20U)，![X\in R^{W'\times H'\times C'}](https://latex.csdn.net/eq?X%5Cin%20R%5E%7BW%27%5Ctimes%20H%27%5Ctimes%20C%27%7D)![U\in R^{W\times H\times C}](https://latex.csdn.net/eq?U%5Cin%20R%5E%7BW%5Ctimes%20H%5Ctimes%20C%7D),(例如卷积或一组卷积)，我们可以构造一个相应的SE块来执行特征重新校准，如下所示。特征U首先通过squeeze操作，该操作跨越空间维度W×H聚合特征映射来产生通道描述符。这个描述符嵌入了通道特征响应的全局分布，使来自网络全局感受野的信息能够被其较低层利用。这之后是一个excitation操作，其中通过基于通道依赖性的自门机制为每个通道学习特定采样的激活，控制每个通道的激励。然后特征映射U被重新加权以生成SE块的输出，然后可以将其直接输入到随后的层中。

![](https://i-blog.csdnimg.cn/blog_migrate/0e655b1150c0ac81d347687fe1355d54.png)

图1. Squeeze-and-Excitation块

SE网络可以通过简单地堆叠SE构建块的集合来生成。SE块也可以用作架构中任意深度的原始块的直接替换。然而，虽然构建块的模板是通用的，正如我们6.3节中展示的那样，但它在不同深度的作用适应于网络的需求。在前面的层中，它学习以类不可知的方式激发信息特征，增强共享的较低层表示的质量。在后面的层中，SE块越来越专业化，并以高度类特定的方式响应不同的输入。因此，SE块进行特征重新校准的好处可以通过整个网络进行累积。

新CNN架构的开发是一项具有挑战性的工程任务，通常涉及许多新的超参数和层配置的选择。相比之下，上面概述的SE块的设计是简单的，并且可以直接与现有的最新架构一起使用，其卷积层可以通过直接用对应的SE层来替换从而进行加强。另外，如第四节所示，SE块在计算上是轻量级的，并且在模型复杂性和计算负担方面仅稍微增加。为了支持这些声明，我们开发了一些SENets，即SE-ResNet，SE-Inception，SE-ResNeXt和SE-Inception-ResNet，并在ImageNet 2012数据集\[30\]上对SENets进行了广泛的评估。此外，为了证明SE块的一般适用性，我们还呈现了ImageNet之外的结果，表明所提出的方法不受限于特定的数据集或任务。

使用SENets，我们赢得了ILSVRC 2017分类竞赛的第一名。我们的表现最好的模型集合在测试集上达到了2.251%的`top-5`错误率。与前一年的获奖者（2.991%的`top-5`错误率）相比，这表示25%的相对改进。我们的模型和相关材料已经提供给研究界。

### 精读 

#### 计算机视觉的研究中心主题 

寻找更强大的特征表示，这种特征表示只捕捉给定任务中最显著的图像属性，从而提高性能。

#### 灵感来源 

最近研究表明，通过将学习机制整合到有助于捕捉特征间空间相关性的网络中，CNN产生的表征可以得到加强。（例：Inception架构将多尺度处理合并到网络模块以提升性能）

进一步工作，寻求更好的模型空间依赖，并将空间注意力融入网络结构。

#### 本文工作 

（1）研究了网络设计的一个不同方面——通道之间的关系。

（2）设计一种新的架构单元——“Squeeze-and-Excitation” (SE)块，其目标是通过显式建模卷积特征通道之间的相互依赖来提高网络产生的表征的质量。（相当于特征做了提纯，使得特征表达能力更强）

（3）本文提出了一种机制，允许网络进行特征重新校准，通过该机制网络可以学习使用全局信息，选择性地强调信息特征和抑制无用的特征

#### SE模块特点 

（1）简单地堆叠一组SE块就可以构建一个SE网络(SENet)。

（2）SE块可用作网络架构中一定深度的原始块的替代品。

（3）SE模块的模板是通用的，但SE模块在整个网络的不同深度所扮演的角色不同。

（4）SE模块的结构很简单。

（5）SE块在计算上是轻量级的。

## 二、Related Work—相关工作 

### Deep architectures—更深架构 

### 翻译 

深层架构。大量的工作已经表明，以易于学习深度特征的方式重构卷积神经网络的架构可以大大提高性能。VGGNets\[35\]和Inception模型\[39\]证明了深度增加可以获得的好处，明显超过了ILSVRC 2014之前的方法。批标准化（BN）\[14\]通过插入单元来调节层输入稳定学习过程，改善了通过深度网络的梯度传播，这使得可以用更深的深度进行进一步的实验。He等人\[9,10\]表明，通过重构架构来训练更深层次的网络是有效的，通过使用基于恒等映射的跳跃连接来学习残差函数，从而减少跨单元的信息流动。最近，网络层间连接的重新表示\[5,12\]已被证明可以进一步改善深度网络的学习和表征属性。

另一种研究方法探索了调整网络模块化组件功能形式的方法。可以用分组卷积来增加基数（一组变换的大小）\[13,43\]以学习更丰富的表示。多分支卷积可以解释为这个概念的概括，使得卷积算子可以更灵活的组合\[14,38,39,40\]。跨通道相关性通常被映射为新的特征组合，或者独立的空间结构\[6,18\]，或者联合使用标准卷积滤波器\[22\]和1×1卷积，然而大部分工作的目标是集中在减少模型和计算复杂度上面。这种方法反映了一个假设，即通道关系可以被表述为具有局部感受野的实例不可知的函数的组合。相比之下，我们声称为网络提供一种机制来显式建模通道之间的动态、非线性依赖关系，使用全局信息可以减轻学习过程，并且显著增强网络的表示能力。

### 精读 

#### 网络深度的研究 

增加网络深度可以显著提高其学习的表征质量（VGG和Inception）

通过调节输入到每一层的分布，增加了深度网络学习过程的稳定性、产生更平滑的优化表面。(BN)

通过短路连接 ，可以学习相当深和强大的网络。（ResNet）

引入了一种门控机制来调节沿短路连接的信息流。（Highway networks）

#### 网络中计算元素的函数形式的研究 

分组卷积： 已被证明是一种流行的增加学习转换基数的方法。

多分支卷积： 可以实现更灵活的算子组合，可以看作是分组操作符的自然扩展。

#### 启发 

为单元提供一种机制，使用全局信息对通道之间的动态、非线性依赖关系进行显式建模，可以简化学习过程，并显著增强网络的表示能力。

### Attention and gating mechanisms—注意力和门控机制 

### 翻译 

注意力和门机制。从广义上讲，可以将注意力视为一种工具，将可用处理资源的分配偏向于输入信号的信息最丰富的组成部分。这种机制的发展和理解一直是神经科学社区的一个长期研究领域\[15,16,28\]，并且近年来作为一个强大补充，已经引起了深度神经网络的极大兴趣\[20,25\]。注意力已经被证明可以改善一系列任务的性能，从图像的定位和理解\[3,17\]到基于序列的模型\[2,24\]。它通常结合门功能（例如softmax或sigmoid）和序列技术来实现\[11,37\]。最近的研究表明，它适用于像图像标题\[4,44\]和口头阅读\[7\]等任务，其中利用它来有效地汇集多模态数据。在这些应用中，它通常用在表示较高级别抽象的一个或多个层的顶部，以用于模态之间的适应。高速网络\[36\]采用门机制来调节快捷连接，使得可以学习非常深的架构。王等人\[42\]受到语义分割成功的启发，引入了一个使用沙漏模块\[27\]的强大的trunk-and-mask注意力机制。这个高容量的单元被插入到中间阶段之间的深度残差网络中。相比之下，我们提出的SE块是一个轻量级的门机制，专门用于以计算有效的方式对通道关系进行建模，并设计用于增强整个网络中模块的表示能力。

### 精读 

注意力：可以被解释为一种将可用的计算资源分配偏向于信号中信息最有用的部分的方法。

> Q：什么是注意力机制？
> 
> 在神经网络学习中，一般而言模型的参数越多则模型的表达能力越强，模型所存储的信息量也越大，但这会带来信息过载的问题。那么通过引入注意力机制，在众多的输入信息中聚焦于对当前任务更为关键的信息，降低对其他信息的关注度，甚至过滤掉无关信息，就可以解决信息过载问题，并提高任务处理的效率和准确性。
> 
> 举个例子：我们人类第一眼看到某个区域的过程其实就是一个注意力机制的应用过程，他可以让我们的大脑更加集中在该区域，并投入更多的关注。

本文SE模块包含一个轻量级的门控机制，该机制通过以计算效率高的方式建模通道关系来增强网络的表示能力。

在目前的神经网络模型中，可以将max pooling和gating机制近似地看作是自下而上的基于显著性的注意力机制。

## 三、Squeeze-and-Excitation Blocks—挤压和激励块 

![](https://i-blog.csdnimg.cn/blog_migrate/077c90022ed4f69f67c6786e91606eac.jpeg)

### 3.1 Transformation（Ftr）—传统卷积操作 

### 翻译 

Squeeze-and-Excitation块是一个计算单元，可以为任何给定的变换构建：![F_{tr}:X\rightarrow U](https://latex.csdn.net/eq?F_%7Btr%7D%3AX%5Crightarrow%20U)，![X\in R^{W'\times H'\times C'}](https://latex.csdn.net/eq?X%5Cin%20R%5E%7BW%27%5Ctimes%20H%27%5Ctimes%20C%27%7D)，![U\in R^{W\times H\times C}](https://latex.csdn.net/eq?U%5Cin%20R%5E%7BW%5Ctimes%20H%5Ctimes%20C%7D)。为了简化说明，在接下来的表示中，我们将FtrFtr看作一个标准的卷积算子。V=\[V1,V2,…,Vc\]表示学习到的一组滤波器核，Vc指的是第c个滤波器的参数。然后我们可以将FtrFtr的输出写作U=\[U1,U2,…,Uc\]，其中![U_{c}=V_{c}*X=\sum_{s=1}^{C'}V_{c}^{s}*X^{s}](https://latex.csdn.net/eq?U_%7Bc%7D%3DV_%7Bc%7D*X%3D%5Csum_%7Bs%3D1%7D%5E%7BC%27%7DV_%7Bc%7D%5E%7Bs%7D*X%5E%7Bs%7D)

这里 ∗表示卷积，![V_{c}=[V_{c}^{1},V_{c}^{2}...V_{c}^{c'}]](https://latex.csdn.net/eq?V_%7Bc%7D%3D%5BV_%7Bc%7D%5E%7B1%7D%2CV_%7Bc%7D%5E%7B2%7D...V_%7Bc%7D%5E%7Bc%27%7D%5D)，![X_{c}=[X^{1},X^{2}...X^{c'}]](https://latex.csdn.net/eq?X_%7Bc%7D%3D%5BX%5E%7B1%7D%2CX%5E%7B2%7D...X%5E%7Bc%27%7D%5D)（为了简洁表示，忽略偏置项）。这里 ![V_{c}^{s}](https://latex.csdn.net/eq?V_%7Bc%7D%5E%7Bs%7D)是 2D空间核，因此表示![V_{c}](https://latex.csdn.net/eq?V_%7Bc%7D)的一个单通道，作用于对应的通道 X。由于输出是通过所有通道的和来产生的，所以通道依赖性被隐式地嵌入到![V_{c}](https://latex.csdn.net/eq?V_%7Bc%7D)中，但是这些依赖性与滤波器捕获的空间相关性纠缠在一起。我们的目标是确保能够提高网络对信息特征的敏感度，以便后续转换可以利用这些功能，并抑制不太有用的功能。我们建议通过显式建模通道依赖性来实现这一点，以便在进入下一个转换之前通过两步重新校准滤波器响应，两步为： squeeze和 excitation。

### 精读 

![](https://i-blog.csdnimg.cn/blog_migrate/195aa4adb91ab1e188ca667c67321e99.png)

 *  输入： X∈R^(H′×W′×C′)
 *  输出： U∈R^(H×W×C)
 *  滤波器集合V=\[V1,V2 ,...,Vc\]： 前面学习到的滤波器参数，Vc表示第c个滤波器参数
 *  ∗ ： 卷积操作
 *  V cs ： 二维空间核，作用于输入X对应通道的卷积核
 *  s： 输入的个数

对于一个C×W×H的输入X，在经过Ftr卷积操作之后，得到的输出Uc（也就是C个大小为H×W的特征图）。

### 3.2 Squeeze: Global Information Embedding（Fsq）—压缩：全局信息嵌入 

### 翻译 

为了解决利用通道依赖性的问题，我们首先考虑输出特征中每个通道的信号。每个学习到的滤波器都对局部感受野进行操作，因此变换输出U的每个单元都无法利用该区域之外的上下文信息。在网络较低的层次上其感受野尺寸很小，这个问题变得更严重。

讨论。转换输出U可以被解释为局部描述子的集合，这些描述子的统计信息对于整个图像来说是有表现力的。特征工程工作中\[31,34,45\]普遍使用这些信息。我们选择最简单的全局平均池化，同时也可以采用更复杂的汇聚策略。

### 精读 

#### 通道依赖问题 

每一个学习到的卷积核，有一个局部感受野，因此每个卷积单元只能够关注到它所在区域内的空间信息，这样的话，这个感受野区域之外的信息就无法利用，卷积核只是在一个局部空间内进行操作， 输出特征图就很难获得足够的信息来提取通道之间的关系。

![](https://i-blog.csdnimg.cn/blog_migrate/07f594de7503f96569ac686622305d31.png)

####  Squeeze操作 

Fsq操作就是使用通道的全局平均池化，将包含全局信息的W×H×C 的特征图直接压缩成一个1×1×C的特征向量，即将每个二维通道变成一个具有全局感受野的数值，此时1个像素表示1个通道，屏蔽掉空间上的分布信息，更好的利用通道间的相关性。

输出的维度和输入的特征通道数相匹配（C）

Zc ： 经Squeeze操作对Uc进行全局平均池化（GAP）得到的特征信息分布的局部描述算子，可以理解为在该层得到的C个特征图的数值分布情况。

### 3.3 Excitation: Adaptive Recalibration（Fex）—激励：自适应重新校准 

### 翻译 

为了利用压缩操作中汇聚的信息，我们接下来通过第二个操作来全面捕获通道依赖性。为了实现这个目标，这个功能必须符合两个标准：第一，它必须是灵活的（特别是它必须能够学习通道之间的非线性交互）；第二，它必须学习一个非互斥的关系，因为独热激活相反，这里允许强调多个通道。为了满足这些标准，我们选择采用一个简单的门机制，并使用sigmoid激活：

![s=F_{ex}(z,W)=\sigma (g(z,W))=\sigma (W_{2}\sigma (W_{1}z))](https://latex.csdn.net/eq?s%3DF_%7Bex%7D%28z%2CW%29%3D%5Csigma%20%28g%28z%2CW%29%29%3D%5Csigma%20%28W_%7B2%7D%5Csigma%20%28W_%7B1%7Dz%29%29)

，其中 ![\sigma](https://latex.csdn.net/eq?%5Csigma)是指ReLU\[26\]函数，![W_{1}\in R^{\frac{C}{r}}\times C](https://latex.csdn.net/eq?W_%7B1%7D%5Cin%20R%5E%7B%5Cfrac%7BC%7D%7Br%7D%7D%5Ctimes%20C)和 ![W_{2}\in R^{\frac{C}{r}}\times C](https://latex.csdn.net/eq?W_%7B2%7D%5Cin%20R%5E%7B%5Cfrac%7BC%7D%7Br%7D%7D%5Ctimes%20C)。为了限制模型复杂度和辅助泛化，我们通过在非线性周围形成两个全连接（FC）层的瓶颈来参数化门机制，即降维层参数为 ![W_{1}](https://latex.csdn.net/eq?W_%7B1%7D)，降维比例为 r（我们把它设置为16，这个参数选择在6.3节中讨论），一个ReLU，然后是一个参数为![W_{2}](https://latex.csdn.net/eq?W_%7B2%7D)的升维层。块的最终输出通过重新调节带有激活的变换输出 U得到：

![\widetilde{X_{c}}=F_{scale}(u_{c},s_{c})=s_{c}\cdot u_{c}](https://latex.csdn.net/eq?%5Cwidetilde%7BX_%7Bc%7D%7D%3DF_%7Bscale%7D%28u_%7Bc%7D%2Cs_%7Bc%7D%29%3Ds_%7Bc%7D%5Ccdot%20u_%7Bc%7D)

其中 ![\widetilde{X}=[\widetilde{x_{1}},\widetilde{x_{2}}...\widetilde{x_{c}}]](https://latex.csdn.net/eq?%5Cwidetilde%7BX%7D%3D%5B%5Cwidetilde%7Bx_%7B1%7D%7D%2C%5Cwidetilde%7Bx_%7B2%7D%7D...%5Cwidetilde%7Bx_%7Bc%7D%7D%5D)和![F_{scale}(u_{c},s_{c})](https://latex.csdn.net/eq?F_%7Bscale%7D%28u_%7Bc%7D%2Cs_%7Bc%7D%29) 指的是特征映射![U_{c}\in R^{^{H\times W}}](https://latex.csdn.net/eq?U_%7Bc%7D%5Cin%20R%5E%7B%5E%7BH%5Ctimes%20W%7D%7D)和标量 ![s_{c}](https://latex.csdn.net/eq?s_%7Bc%7D)之间的对应通道乘积。

讨论。激活作为适应特定输入描述符z的通道权重。在这方面，SE块本质上引入了以输入为条件的动态特性，有助于提高特征辨别力。

### 精读 

目的:为了利用在"squeeze"操作中聚合的信息，接着进行Excitation操作，来完全捕获通道依赖关系。

方法：为实现上述目标，函数必须符合两个标准：

(1）灵活性：它必须能够学习通道之间的非线性相互作用

(2）必须学习一种非互斥关系：因为我们希望确保允许强调多个通道不同重要程度(而不是强制一个one-hot激活)。因为我们不光要学习特征，还要学习通道之间信息的相关性。

为了满足这两个条件，这篇论文这里采用两个全连接层+两个激活函数组成的结构输出和输入特征同样数目的权重值，也就是每个特征通道的权重系数。

![](https://i-blog.csdnimg.cn/blog_migrate/97caf9a9b930d71d88aef28b95ddc39c.png)

####  Excitation操作 

基于特征通道间的相关性，每个特征通道生成一个权重，用来代表特征通道的重要程度。由原本全为白色的C个通道的特征，得到带有不同深浅程度的颜色的特征向量，也就是不同的重要程度。

(1)第一个FC层：ReLU （δ）

(2)第二个FC层：Sigmoid（σ）

> Q1：加入全连接层的作用？
> 
> 这是为了利用通道间的相关性来训练出真正的scale。一次mini-batch个样本的squeeze输出并不代表通道真实要调整的scale值，真实的scale要基于全部数据集来训练得出，而不是基于单个batch，所以后面要加个全连接层来进行训练。

> Q2：为什么加两个全连接层？
> 
> 应该是类似bottleneck的设计，增加非线性（model capacity)，减少参数和运算量，不压缩的话这块儿的参数量和运算量会多r^2倍，比如1024个特征到1024个特征，直接全连接运算量是1024×1024，如果中间插入一个256层，那么它的运算量是1024×256×2，运算量降低了一半。

> Q3：为什么前面用ReLU激活，后面为什么要改用Sigmoid呢？
> 
> (1)具有更多的非线性：可以更好地拟合通道间复杂的相关性。
> 
> (2)极大地减少了参数量和计算量：降维参数 r 用于控制第一个FC层中的神经元个数，在论文中也是经过多次对比实验得出r=16 时，模型得到的效果是最好的。
> 
> (3)由于Sigmoid函数图像的特点，它的值域在0—1之间，那么这样很符合概率分布的特点，最后能够获得 在0—1 之间归一化的权重参数，这样的话再通过乘法逐通道加权到先前的特征图上，使得有用的信息的注意力更趋向于1，而没有用的信息则更趋向于0，得到最后带有注意力权重的特征图。

公式理解：输入 z 经过两个FC层加激活函数，最终得到权重参数s 。（这个得到的 s 其实才是本文的核心，才是注意力机制的体现，它是用来刻画输出U中 C 个feature map 的权重，而且这个权重是通过前面这些全连接层和非线性层学习得到的，因此可以进行端到端训练。这两个全连接层的作用就是融合前面Squeeze针对各通道操作得到的特征信息。 ）

### 3.4 Scale：Reweight（Fscale ）—权重 

### 翻译 

最后是Scale操作，将前面得到的注意力权重加权到每个通道的特征上。论文中采用乘法，逐通道乘以权重系数，在通道维度上引入attention机制，也就是把当前得到的带有注意力的权重参数渲染到原来的特征图U上，得到![\widetilde{X}](https://latex.csdn.net/eq?%5Cwidetilde%7BX%7D)，将原本全为白色的注意力权重变为不同颜色的状态，也就代表着不同程度的重要性，对于某些颜色较深的attention map，神经网络就会重点关注这些特征通道，也即提升对当前任务有用的特征通道，并且抑制对当前任务用处不大的特征通道，这样就对原始特征进行了重标定。

### 精读 

将原本全为白色的注意力权重变为不同颜色的状态，也就代表着不同程度的重要性，对于某些颜色较深的attention map，神经网络就会重点关注这些特征通道。

![](https://i-blog.csdnimg.cn/blog_migrate/76f52c0ebfe0cdcf7092fe6b1e1fd29f.png)

####  Reweight操作 

将Excitation输出的权重看做每个特征通道的重要性，也就是对于U每个位置上的所有H×W上的值都乘上对应通道的权值，完成对原始特征的重校准。

公式 Fscale(uc ,sc )即为特征映射uc和标量sc（这里由于是1×1可看做标量）之间的对应通道乘积操作。

### 3.5 Instantiations—实例 

### 翻译 

SE块可以通过在每次卷积之后的非线性之后插入而集成到标准体系结构中，例如VGGNet \[11\]。 此外，SE块的灵活性意味着它可以直接应用于标准卷积以外的转换。 为了说明这一点，我们通过将SE块合并到一些更复杂的体系结构示例中来开发SENet，如下所述。

我们首先考虑为接收网络构建SE块\[5\]。 在这里，我们仅将变换Ftr当作一个完整的Inception模块（参见图2），并通过对体系结构中的每个此类模块进行此更改，就可以获得SE-Inception网络。 SE块也可以直接用于剩余网络（图3描绘了SE-ResNet模块的架构）。 在此，SE块变换Ftr被认为是残差模块的非同一性分支。 挤压和激励在与身份分支求和之前都起作用。 可以通过类似的方案来构建将SE块与ResNeXt \[19\]，Inception-ResNet \[21\]，MobileNet \[64\]和ShuffleNet \[65\]集成在一起的其他变体。 对于SENet体系结构的具体示例，表1中给出了SE-ResNet-50和SE-ResNeXt-50的详细说明。

SE块具有灵活性的结果之一是，有几种可行的方法可以将其集成到这些体系结构中。 因此，为了评估对用于将SE块合并到网络体系结构中的集成策略的敏感性，我们还提供了消融实验，以探索第6.5节中针对块包含的不同设计。

### 精读 

设计一个新的有效的CNN是很困难的，SE Block 是不改变原来的CNN结构，做到即插即用的，可以将SE块集成到很多网络结构中，比如VGG，甚至是一些更复杂的结构如Inception、ResNet等。

#### （1）SEInception 

将转换Ftr作为一个完整的Inception模块，通过对架构中的每个这样的模块进行更改，获得了一个SE-Inception网络。

![](https://i-blog.csdnimg.cn/blog_migrate/66f79c021f4990e9828e2d58bb5770f1.png)

 SE块详细过程

1.首先由 Inception结构 或 ResNet结构处理后的C×W×H特征图开始，通过Squeeze操作对特征图进行全局平均池化（GAP），得到1×1×C 的特征向量

2.紧接着两个 FC 层组成一个 Bottleneck 结构去建模通道间的相关性：

经过第一个FC层，将C个通道变成 C/ r ，减少参数量，然后通过ReLU的非线性激活，到达第二个FC层

经过第二个FC层，再将特征通道数恢复到C个，得到带有注意力机制的权重参数

3.最后经过Sigmoid激活函数，最后通过一个 Scale 的操作来将归一化后的权重加权到每个通道的特征上。

#### （2）SEResNet 

SE模块可以直接与ResNet一起使用。SE块变换Ftr被认为是残差模块的非恒等分支。挤压和激励都是在同分支相加之前起作用的。

![](https://i-blog.csdnimg.cn/blog_migrate/245869928c3168b5cd8606324ea995a4.png)

#### （3）SE和其他网络集成 

将SE块与ResNeXt、Inception-ResNet、MobileNet和ShuffleNet整合的进一步变体可以通过类似的方案构建。

![](https://i-blog.csdnimg.cn/blog_migrate/8646f580304a4efc33ed40966b40a7aa.png)

## 四、Model and Computational Complexity—模型和计算复杂度 

### 4.1 计算复杂度 

### 翻译 

SENet通过堆叠一组SE块来构建。实际上，它是通过用原始块的SE对应部分（即SE残差块）替换每个原始块（即残差块）而产生的。我们在表1中描述了SE-ResNet-50和SE-ResNeXt-50的架构。

![](https://i-blog.csdnimg.cn/blog_migrate/831b57b8775cb67163a776932ed099ca.png)

表1。(左)ResNet-50，(中)SE-ResNet-50，(右)具有32×4d模板的SE-ResNeXt-50。在括号内列出了残差构建块特定参数设置的形状和操作，并且在外部呈现了一个阶段中堆叠块的数量。fc后面的内括号表示SE模块中两个全连接层的输出维度。

在实践中提出的SE块是可行的，它必须提供可接受的模型复杂度和计算开销，这对于可伸缩性是重要的。为了说明模块的成本，作为例子我们比较了ResNet-50和SE-ResNet-50，其中SE-ResNet-50的精确度明显优于ResNet-50，接近更深的ResNet-101网络（如表2所示）。对于224×224像素的输入图像，ResNet-50单次前向传播需要 3.86 GFLOP。每个SE块利用压缩阶段的全局平均池化操作和激励阶段中的两个小的全连接层，接下来是廉价的通道缩放操作。总的来说，SE-ResNet-50需要 3.87 GFLOP，相对于原始的ResNet-50只相对增加了0.26%。

![Table 2](https://i-blog.csdnimg.cn/blog_migrate/3bf7bfbe2ecb81852a70a6e8df4b4167.png)

表2。ImageNet验证集上的单裁剪图像错误率（％）和复杂度比较。`original`列是指原始论文中报告的结果。为了进行公平比较，我们重新训练了基准模型，并在`re-implementation`列中报告分数。`SENet`列是指已添加SE块后对应的架构。括号内的数字表示与重新实现的基准数据相比的性能改善。†表示该模型已经在验证集的非黑名单子集上进行了评估（在\[38\]中有更详细的讨论），这可能稍微改善结果。

在实践中，训练的批数据大小为256张图像，ResNet-50的一次前向传播和反向传播花费190 ms，而SE-ResNet-50则花费209 ms（两个时间都在具有8个NVIDIA Titan X GPU的服务器上执行）。我们认为这是一个合理的开销，因为在现有的GPU库中，全局池化和小型内积操作的优化程度较低。此外，由于其对嵌入式设备应用的重要性，我们还对每个模型的CPU推断时间进行了基准测试：对于224×224像素的输入图像，ResNet-50花费了164ms，相比之下，SE-ResNet-50花费了167ms。SE块所需的小的额外计算开销对于其对模型性能的贡献来说是合理的（在第6节中详细讨论）。

### 精读 

（1）将SE模块嵌入到复杂结构的模型上（如 VGG、Inception）

（2）将SE模块嵌入到轻量化模型（如 MobileNet、ShuffleNet）

#### 结论 

（1）SE-Module均带来了不同程度的性能提升

（2）SE-ResNet-50精度与ResNet-101差不多，但是GFLOPs少了一半

（3）SE-Module与网络深度带来的提升是互补的

### 4.2 参数量 

### 翻译 

接下来，我们考虑所提出的块引入的附加参数。所有附加参数都包含在门机制的两个全连接层中，构成网络总容量的一小部分。更确切地说，引入的附加参数的数量由下式给出：

![\frac{2}{r}\sum_{s=1}^{s}N_{s}\cdot C_{s}^{2}](https://latex.csdn.net/eq?%5Cfrac%7B2%7D%7Br%7D%5Csum_%7Bs%3D1%7D%5E%7Bs%7DN_%7Bs%7D%5Ccdot%20C_%7Bs%7D%5E%7B2%7D)

其中 r表示减少比率（我们在所有的实验中将r设置为 16）， S指的是阶段数量（每个阶段是指在共同的空间维度的特征映射上运行的块的集合），![C_{s}](https://latex.csdn.net/eq?C_%7Bs%7D)表示阶段 s的输出通道的维度，![N_{s}](https://latex.csdn.net/eq?N_%7Bs%7D)表示重复的块编号。总的来说，SE-ResNet-50在ResNet-50所要求的 2500万参数之外引入了 250万附加参数，相对增加了 10%的参数总数量。这些附加参数中的大部分来自于网络的最后阶段，其中激励在最大的通道维度上执行。然而，我们发现SE块相对昂贵的最终阶段可以在性能的边际成本（ImageNet数据集上 <0.1 \\%的top-1错误率）上被移除，将相对参数增加减少到 4%，这在参数使用是关键考虑的情况下可能证明是有用的。

### 精读 

由于SE块中的操作主要的附加参数都来自于Excitation操作中门控机制的两个全连接层，具体来分析，这些FC层的权重参数引入的总数为：

![](https://i-blog.csdnimg.cn/blog_migrate/a3803f20f846613de50d747991be2f8a.png)

#### 两个FC层中特征维度的变化： 

一个Block块中的参数量增加为 2C^2/r

一个Stage阶段中有N个Block

一个模型中有S个Stage

#### 公式理解： 

r： 缩减比率reduction ratio r，控制第一个FC层神经元个数，通常为16，r越大，带来的参数量就越少

S： 表示阶段stage，不同stage分开计算，因为不同stage特征通道数是不一样的

Ns ： 第S个stage有多少个Building Block重复堆叠

Cs ： 第S个stage的Building Block中的特征图有多少个通道（输出通道维度）

## 五、Implementation—实施 

### 翻译 

在训练过程中，我们遵循标准的做法，使用随机大小裁剪\[39\]到224×224像素（299×299用于Inception-ResNet-v2\[38\]和SE-Inception-ResNet-v2）和随机的水平翻转进行数据增强。输入图像通过通道减去均值进行归一化。另外，我们采用\[32\]中描述的数据均衡策略进行小批量采样，以补偿类别的不均匀分布。网络在我们的分布式学习系统“ROCS”上进行训练，能够处理大型网络的高效并行训练。使用同步SGD进行优化，动量为0.9，小批量数据的大小为1024（在4个服务器的每个GPU上分成32张图像的子批次，每个服务器包含8个GPU）。初始学习率设为0.6，每30个迭代周期减少10倍。使用\[8\]中描述的权重初始化策略，所有模型都从零开始训练100个迭代周期。

### 精读 

数据集：ImageNet 2012

训练设置：

 *  随机的水平翻转
 *  输入图像平均rgb通道减法进行归一化
 *  所有模型都在分布式学习系统ROCS上进行训练
 *  优化器使用同步SGD，动量为0.9，小批量数据的大小为1024（在4个服务器的每个GPU上分成32张图像的子批次，每个服务器包含8个GPU）
 *  初始学习率设为0.6
 *  每30个迭代周期减少10倍
 *  所有模型都从零开始训练100个迭代周期

评估设置：

使用随机大小裁剪到224×224像素（299×299用于Inception-ResNet-v2和SE-Inception-ResNet-v2）

## 六、Experiments—实验 

### 6.1 Image Classificatio—图像分类 

### 翻译 

网络深度。我们首先将SE-ResNet与一系列标准ResNet架构进行比较。每个ResNet及其相应的SE-ResNet都使用相同的优化方案进行训练。验证集上不同网络的性能如表2所示，表明SE块在不同深度上的网络上计算复杂度极小增加，始终提高性能。

值得注意的是，SE-ResNet-50实现了单裁剪图像6.62%的`top-5`验证错误率，超过了ResNet-50（7.48%）0.86%，接近更深的ResNet-101网络（6.52%的`top-5`错误率），且只有ResNet-101一半的计算开销（3.87 GFLOPs vs. 7.58GFLOPs）。这种模式在更大的深度上重复，SE-ResNet-101（6.07%的`top-5`错误率）不仅可以匹配，而且超过了更深的ResNet-152网络（6.34%的`top-5`错误率）。图4分别描绘了SE-ResNets和ResNets的训练和验证曲线。虽然应该注意SE块本身增加了深度，但是它们的计算效率极高，即使在扩展的基础架构的深度达到收益递减的点上也能产生良好的回报。而且，我们看到通过对各种不同深度的训练，性能改进是一致的，这表明SE块引起的改进可以与增加基础架构更多深度结合使用。

![Figure 4](https://i-blog.csdnimg.cn/blog_migrate/a5f9df2fa20317245fe9d5daf2a85636.png)

图4。ImageNet上的训练曲线。(左)：ResNet-50和SE-ResNet-50；(右)：ResNet-152和SE-ResNet-152。

与现代架构集成。接下来我们将研究SE块与另外两种最先进的架构Inception-ResNet-v2\[38\]和ResNeXt\[43\]的结合效果。Inception架构将卷积模块构造为分解滤波器的多分支组合，反映了Inception假设\[6\]，可以独立映射空间相关性和跨通道相关性。相比之下，ResNeXt体架构断言，可以通过聚合稀疏连接（在通道维度中）卷积特征的组合来获得更丰富的表示。两种方法都在模块中引入了先前结构化的相关性。我们构造了这些网络的SENet等价物，SE-Inception-ResNet-v2和SE-ResNeXt（表1给出了SE-ResNeXt-50（32×4d）的配置）。像前面的实验一样，原始网络和它们对应的SENet网络都使用相同的优化方案。

表2中给出的结果说明在将SE块引入到两种架构中会引起显著的性能改善。尤其是SE-ResNeXt-50的`top-5`错误率是5.49%，优于于它直接对应的ResNeXt-50（5.90%的`top-5`错误率）以及更深的ResNeXt-101（5.57%的`top-5`错误率），这个模型几乎有两倍的参数和计算开销。对于Inception-ResNet-v2的实验，我们猜测可能是裁剪策略的差异导致了其报告结果与我们重新实现的结果之间的差距，因为它们的原始图像大小尚未在\[38\]中澄清，而我们从相对较大的图像（其中较短边被归一化为352）中裁剪出299×299大小的区域。SE-Inception-ResNet-v2（4.79%的`top-5`错误率）比我们重新实现的Inception-ResNet-v2（5.21%的`top-5`错误率）要低0.42%相对改进了8.1%）也优于\[38\]中报告的结果。每个网络的优化曲线如图5所示，说明了在整个训练过程中SE块产生了一致的改进。

![Figure 5](https://i-blog.csdnimg.cn/blog_migrate/58daccda9220514f8843c302ec6ace2f.png)

图5。ImageNet的训练曲线。(左): ResNeXt-50和SE-ResNeXt-50；(右)：Inception-ResNet-v2和SE-Inception-ResNet-v2。

### 精读 

#### Network depth—网络深度 

![](https://i-blog.csdnimg.cn/blog_migrate/eb25c35b8fea1a608b2960866e2a5e52.png)

由上图我们可以得到如下结论：

（1）SE块在不同深度上的网络上计算复杂度极小增加，始终提高性能。

（2）SE块本身增加了深度，但它们以一种极其高效的计算方式这样做，并且即使在扩展基本架构的深度达到收益递减的情况下也会产生良好的回报。

（3）对各种不同深度的训练，性能改进是一致的，这表明SE块引起的改进可以与增加基础架构更多深度结合使用。

#### Integration with modern architectures—与现代架构的融合 

（1）研究将SE模块与另外两种最先进的架构整合的效果：Inception-Resnet-v2和ResNeXt

![](https://i-blog.csdnimg.cn/blog_migrate/91acc6334962cc12971a5c5ba9c1dd4e.png)

 结论： 在两个体系结构中引入SE块带来显著性能改进。

（2）评估SE块在非残差网络上运行时的影响：通过使用VGG16 和BN-Inception架构进行实验 。

![](https://i-blog.csdnimg.cn/blog_migrate/14b403052c345c663523e85eb7713ba3.png)

结论： 这个结果适用于残差和非残差基础。

#### Results on ILSVRC 2017 Classification Competition—ILSVRC 2017分类竞赛的结果 

![](https://i-blog.csdnimg.cn/blog_migrate/5dd5f4abe47bf3733c60da2c42465ed2.png)

SENets在ILSVRC2017竞赛获得了第一名，获奖作品由一小群SENets组成，采用了标准的multi-scale 和 multi-crop融合策略，在测试集上获得了2.251%的top-5误差。

###  6.2 Scene Classification—场景分类 

### 翻译 

我们还对Places365-Challenge数据集\[73\]进行了实验，以进行场景分类。 该数据集包含800万个训练图像和365个类别中的36、500个验证图像。 相对于分类，场景理解的任务为模型的良好概括和处理抽象的能力提供了另一种评估方法。因为它通常要求模型处理更复杂的数据关联，并且对于更大程度的外观变化具有鲁棒性。

我们选择使用ResNet-152作为评估SE区块有效性的强大基准，并遵循\[72\]，\[74\]中所述的训练和评估协议。 在这些实验中，模型是从头开始训练的。 我们将结果报告在表6中，并与先前的工作进行了比较。 我们观察到SE-ResNet-152（11.01％的top-5误差）提供了证据，表明SE块也可以改善场景分类。 该SENet超越了以前的最新模型Places-365-CNN \[72\]，该模型在此任务上的前5个误差为11.48％。

### 精读 

#### Places365-Challenge数据集 

Places365-Challenge数据集上进行了场景分类实验，包含800万张训练图像和36500张验证图像，跨越365个类别。相对于分类，场景理解的任务提供了另一种评估模型的能力，以很好地概括和处理抽象，这是因为它通常要求模型处理更复杂的数据关联，并对更大级别的外观变化具有鲁棒性。

#### 实验设计与结果 

![](https://i-blog.csdnimg.cn/blog_migrate/021f8cc236bd9a6df9ef010b8d1a4119.png)

SE-ResNet-152 (11.01% top-5误差)的验证误差低于ResNet-152 (11.61% top-5误差)，表明SE块也可以改善场景分类，SENet超过了之前最先进的模型 Places-365-CNN（11.48%的 top-5错误）

### 6.3 Object Detection on COCO—关于COCO的对象检测 

### 翻译 

我们进一步使用COCO数据集评估SE块在对象检测任务上的一般化\[75\]。 与以前的工作\[19\]一样，我们使用最小协议，即在80k训练集和35k val子集的联合上训练模型，并对剩余的5k val子集进行评估。 权重由ImageNet数据集上训练的模型的参数初始化。 我们使用Faster R-CNN \[4\]检测框架作为评估模型的基础，并遵循\[76\]中描述的超参数设置（即，采用“ 2x”学习时间表进行端到端训练）。 我们的目标是评估用SE-ResNet替换对象检测器中的中继体系结构（ResNet）的效果，以便性能的任何变化都可以归因于更好的表示形式。

表7报告了使用ResNet-50，ResNet-101及其SE对应物作为中继线体系结构的对象检测器的验证集性能。 在COCO的标准AP指标上，SE-ResNet-50优于ResNet-50 2.4％（相对6.3％的改善），在AP@IoU=0.5时，SE-ResNet-50优于ResNet-50。 SE块还受益于更深的ResNet-101体系结构，在AP指标上实现了2.0％的提高（相对于5.0％的相对改进）。 总之，这组实验证明了SE嵌段的通用性。 可以在广泛的体系结构，任务和数据集上实现改进。

### 精读 

#### 实验设置 

（1）数据集： 在之前的工作中使用minival策略，即在80k的训练集和35k的val子集的并集上训练模型，并在其余5k的val子集上评估。

（2）权重（预训练模型）： 权重由在ImageNet数据集上训练的模型参数初始化。

（3）检测架构：使用Faster R-CNN检测框架，并遵循原文描述的超参数设置。

#### 实验目标 

评估用SE-ResNet替换目标检测器中的主干架构(ResNet)的效果，以便性能上的任何变化都可以归因于更好的表示。

#### 结论 

![](https://i-blog.csdnimg.cn/blog_migrate/192634b7651971550e39b86880566ccb.png)

实验证明了SE模块的普遍性。改进可以在广泛的架构、任务和数据集上实现。

### 6.4 Analysis and Interpretation—分析和讨论 

### 翻译 

减少比率。公式（5）中引入的减少比率r是一个重要的超参数，它允许我们改变模型中SE块的容量和计算成本。为了研究这种关系，我们基于SE-ResNet-50架构进行了一系列不同r值的实验。表5中的比较表明，性能并没有随着容量的增加而单调上升。这可能是使SE块能够过度拟合训练集通道依赖性的结果。尤其是我们发现设置r=16在精度和复杂度之间取得了很好的平衡，因此我们将这个值用于所有的实验。

![Table 5](https://i-blog.csdnimg.cn/blog_migrate/1585a4c2a80d730e83d539aff2a991ae.png)

表5。 ImageNet验证集上单裁剪图像的错误率(%)和SE-ResNet-50架构在不同减少比率rr下的模型大小。这里`original`指的是ResNet-50。

激励的作用。虽然SE块从经验上显示出其可以改善网络性能，但我们也想了解自门激励机制在实践中是如何运作的。为了更清楚地描述SE块的行为，本节我们研究SE-ResNet-50模型的样本激活，并考察它们在不同块不同类别下的分布情况。具体而言，我们从ImageNet数据集中抽取了四个类，这些类表现出语义和外观多样性，即金鱼，哈巴狗，刨和悬崖（图7中显示了这些类别的示例图像）。然后，我们从验证集中为每个类抽取50个样本，并计算每个阶段最后的SE块中50个均匀采样通道的平均激活（紧接在下采样之前），并在图8中绘制它们的分布。作为参考，我们也绘制所有1000个类的平均激活分布。

![Figure 7](https://i-blog.csdnimg.cn/blog_migrate/62a83d396d9d4ceef1dbf9ffca35473d.png)

图7。ImageNet中四个类别的示例图像。

![Figure 8](https://i-blog.csdnimg.cn/blog_migrate/111aa5ac362243fe8663156e41745540.png)

图8。SE-ResNet-50不同模块在ImageNet上由Excitation引起的激活。模块名为“SE stageID blockID”。

我们对SENets中Excitation的作用提出以下三点看法。首先，不同类别的分布在较低层中几乎相同，例如，SE\_2\_3。这表明在网络的最初阶段特征通道的重要性很可能由不同的类别共享。然而有趣的是，第二个观察结果是在更大的深度，每个通道的值变得更具类别特定性，因为不同类别对特征的判别性值具有不同的偏好。SE\_4\_6和SE\_5\_1。这两个观察结果与以前的研究结果一致\[21,46\]，即低层特征通常更普遍（即分类中不可知的类别），而高层特征具有更高的特异性。因此，表示学习从SE块引起的重新校准中受益，其自适应地促进特征提取和专业化到所需要的程度。最后，我们在网络的最后阶段观察到一个有些不同的现象。SE\_5\_2呈现出朝向饱和状态的有趣趋势，其中大部分激活接近于1，其余激活接近于0。在所有激活值取1的点处，该块将成为标准残差块。在网络的末端SE\_5\_3中（在分类器之前紧接着是全局池化），类似的模式出现在不同的类别上，尺度上只有轻微的变化（可以通过分类器来调整）。这表明，SE\_5\_2和SE\_5\_3在为网络提供重新校准方面比前面的块更不重要。这一发现与第四节实证研究的结果是一致的，这表明，通过删除最后一个阶段的SE块，总体参数数量可以显著减少，性能只有一点损失（<0.1%的`top-1`错误率）。

### 精读 

#### Reduction ratio—衰减率 

衰减率r是一个超参数，它允许我们改变网络中SE块的容量和计算成本。为了研究性能和计算成本之间的平衡，使用SE-ResNet-50在不同的r值范围内进行实验，如表7

![](https://i-blog.csdnimg.cn/blog_migrate/728b815f663b622e1de7408c251bdb37.png)

#### 结论 

（1）性能在一定范围的衰减率下是稳健的。

（2）性能并没有随着r的增加而单调上升，设置r = 16可以很好地平衡准确性和复杂性。

#### The role of Excitation—激励的作用 

图8：SE-ResNet-50不同模块在ImageNet上由Excitation引起的激活。模块名为“SE stageID blockID”。

![](https://i-blog.csdnimg.cn/blog_migrate/93481db6f6cb80153ec747f6b6141e86.png)

结论

（1）靠前的层级（SE\_2\_3和SE\_3\_4）各个分类的曲线差异不大，这说明了在较低的层级中scale的分布和输入的类别无关

（2）随着层级的加深，不同类别的曲线开始出现了差别（SE\_4\_6和SE\_5\_1），这说明靠后的层级的scale大小和输入的类别强相关

（3）到了SE\_5\_2后几乎所有的scale都饱和，输出为1，只有一个通道为0

（4）而最后一层SE\_5\_3的通道的scale基本相同

（5）最后两层的scale因为基本都相等，就没有什么用处了，为了节省计算量可以把它们去掉

## 七、Conclusion—结论 

### 翻译 

在本文中，我们提出了SE块，这是一种新颖的架构单元，旨在通过使网络能够执行动态通道特征重新校准来提高网络的表示能力。大量实验证明了SENets的有效性，其在多个数据集上取得了最先进的性能。此外，它们还提供了一些关于以前架构在建模通道特征依赖性上的局限性的洞察，我们希望可能证明SENets对其它需要强判别性特征的任务是有用的。最后，由SE块引起的特征重要性可能有助于相关领域，例如为了压缩的网络修剪。

### 精读 

本文提出SE模块，这是一个架构单元，旨在通过使网络能够执行动态通道特征重新校准来提高网络的表征能力。

## 论文十问 

> Q1：论文试图解决什么问题？
> 
> 本文作者研究了网络设计的另一个方面——通道间的关系，进而提出了一种新的结构单元， 命名为：Squeeze-andExcitation (SE) block。能够生成通道维度的权重向量，用于特征“重构”，实现强调重要特征，忽略不重要特征，增强模型表征能力，提高模型性能的目的。

> Q2：这是否是一个新的问题？
> 
> 本文整体上其实没有提出一个新的网络模型，而是相当于提出了一个即插即用的高性能小插件，这是一种新颖的架构单元。

> Q3：这篇文章要验证一个什么科学假设？
> 
> 本篇论文把研究重点放在通道维度上，探索特征通道之间的关系是否能提升网络性能

> Q4：有哪些相关研究？如何归类？谁是这一课题在领域内值得关注的研究员？
> 
> 注意力机制相关研究：STNet、CBAM

> Q5：论文中提到的解决方案之关键是什么？
> 
> 核心思想：增强有用信息，抑制不重要信息
> 
> 核心操作：Squeeze和Excitation

> Q6：论文中的实验是如何设计的？
> 
> 图像分类
> 
> 场景分类
> 
> 关于COCO的对象检测
> 
> 消融实验

> Q7：用于定量评估的数据集是什么？代码有没有开源？
> 
> ImageNet 2012
> 
> Places365-Challenge数据集
> 
> COCO数据集
> 
> 代码有开源

> Q8：论文中的实验及结果有没有很好地支持需要验证的科学假设？
> 
> 证明了。将SE模块加入以前的结构中，均实现了性能的提升，斩获了ILSVRC-2017分类挑战赛的第一

> Q9：这篇论文到底有什么贡献？
> 
> （1）引入了一种叫做SE block的新的结构单元，它能够建模特征图通道间的相互依赖关系，（2）用来增强CNN的表征质量。
> 
> （3）提出了注意力机制。
> 
> （4）该模块可以很容易到嵌入到大多数CNN，任何层次，任何深度，从早期阶段到后期阶段都可以嵌入，即插即用。

> Q10：下一步呢？有什么工作可以继续深入？
> 
> 由SE块引起的特征重要性可能有助于相关领域，例如为了压缩的网络修剪。

本篇到这里就结束了。

代码复现：[SENet代码复现＋超详细注释（PyTorch）][SENet_PyTorch]

下一期：ResNext


[DenseNet]: https://blog.csdn.net/weixin_43334693/article/details/128478420?spm=1001.2014.3001.5501
[SENet_https_arxiv.org_pdf_1709.01507.pdf]: https://arxiv.org/pdf/1709.01507.pdf
[GitHub - hujie-frank_SENet_ Squeeze-and-Excitation Networks]: https://github.com/hujie-frank/SENet
[Link 1]: #%E5%89%8D%E8%A8%80
[Abstract]: #%C2%A0Abstract%E2%80%94%E6%91%98%E8%A6%81
[Introduction]: #%E4%B8%80%E3%80%81Introduction%E2%80%94%E5%BC%95%E8%A8%80
[Related Work]: #%E4%BA%8C%E3%80%81Related%20Work%E2%80%94%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C
[Deep architectures]: #Deep%20architectures%E2%80%94%E6%9B%B4%E6%B7%B1%E6%9E%B6%E6%9E%84
[Attention and gating mechanisms]: #Attention%20and%20gating%20mechanisms%E2%80%94%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%92%8C%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6
[Squeeze-and-Excitation Blocks]: #%E4%B8%89%E3%80%81Squeeze-and-Excitation%20Blocks%E2%80%94%E6%8C%A4%E5%8E%8B%E5%92%8C%E6%BF%80%E5%8A%B1%E5%9D%97
[Transformation_Ftr]: #Transformation%EF%BC%88Ftr%EF%BC%89%E2%80%94%E4%BC%A0%E7%BB%9F%E5%8D%B7%E7%A7%AF%E6%93%8D%E4%BD%9C
[Squeeze_ Global Information Embedding_Fsq]: #Squeeze%3A%20Global%20Information%20Embedding%EF%BC%88Fsq%EF%BC%89%E2%80%94%E5%8E%8B%E7%BC%A9%EF%BC%9A%E5%85%A8%E5%B1%80%E4%BF%A1%E6%81%AF%E5%B5%8C%E5%85%A5
[Excitation_ Adaptive Recalibration_Fex]: #Excitation%3A%20Adaptive%20Recalibration%EF%BC%88Fex%EF%BC%89%E2%80%94%E6%BF%80%E5%8A%B1%EF%BC%9A%E8%87%AA%E9%80%82%E5%BA%94%E9%87%8D%E6%96%B0%E6%A0%A1%E5%87%86
[Scale_Reweight_Fscale]: #Scale%EF%BC%9AReweight%EF%BC%88Fscale%E2%80%8B%20%EF%BC%89%E2%80%94%E6%9D%83%E9%87%8D
[Instantiations]: #Instantiations%E2%80%94%E5%AE%9E%E4%BE%8B
[Model and Computational Complexity]: #%E5%9B%9B%E3%80%81Model%20and%20Computational%20Complexity%E2%80%94%E6%A8%A1%E5%9E%8B%E5%92%8C%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6
[Link 2]: #%E8%AE%A1%E7%AE%97%E5%A4%8D%E6%9D%82%E5%BA%A6
[Link 3]: #%E5%8F%82%E6%95%B0%E9%87%8F
[Implementation]: #%E4%BA%94%E3%80%81Implementation%E2%80%94%E5%AE%9E%E6%96%BD
[Experiments]: #%E5%85%AD%E3%80%81Experiments%E2%80%94%E5%AE%9E%E9%AA%8C
[6.1 Image Classificatio]: #6.1%20Image%20Classificatio%E2%80%94%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB
[6.2 Scene Classification]: #6.2%20Scene%20Classification%E2%80%94%E5%9C%BA%E6%99%AF%E5%88%86%E7%B1%BB
[6.3 Object Detection on COCO_COCO]: #6.3%20Object%20Detection%20on%20COCO%E2%80%94%E5%85%B3%E4%BA%8ECOCO%E7%9A%84%E5%AF%B9%E8%B1%A1%E6%A3%80%E6%B5%8B
[6.4 Analysis and Interpretation]: #6.4%20Analysis%20and%20Interpretation%E2%80%94%E5%88%86%E6%9E%90%E5%92%8C%E8%AE%A8%E8%AE%BA
[Conclusion]: #%E4%B8%83%E3%80%81Conclusion%E2%80%94%E7%BB%93%E8%AE%BA
[Link 4]: #%E8%AE%BA%E6%96%87%E5%8D%81%E9%97%AE
[https_githorb.com_hujie-frank_senet]: http://xn--https-si33a//githorb.com/hujie-frank/senet
[SENet_PyTorch]: https://blog.csdn.net/weixin_43334693/article/details/128567913?spm=1001.2014.3001.5501