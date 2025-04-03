
## 前言 

最近我在研究YOLOv5的改进，一个模型的好坏、改进后效果如何都是需要一系列指标来判断的。这篇就是我将这几天学到的内容做一下总结。

目录

[前言][Link 1]

[🌟一、性能指标][Link 2]

[1.1 混淆矩阵][1.1]

[1.2 Accuracy ：准确率（正确率）][1.2 Accuracy]

[1.3 Precision ：精确率（查准率）][1.3 Precision]

[1.4 Recall ：召回率（查全率）][1.4 Recall]

[1.5 PR曲线][1.5 PR]

[1.6 AP （Average Precision）： 平均精度][1.6 AP _Average Precision_]

[1.7 mAP（mean Average Precision）：均值平均精度][1.7 mAP_mean Average Precision]

[1.8 F1-score与 F值(F-Measure)][1.8 F1-score_ F_F-Measure]

[🌟二、训练结果分析][Link 3]

[2.1 weights ：权重][2.1 weights]

[2.2 confusion\_matrix.png ： 混淆矩阵][2.2 confusion_matrix.png _]

[2.3 F1\_curve.png ：F1曲线][2.3 F1_curve.png _F1]

[2.4 events.out.tfevents：可视化文件][2.4 events.out.tfevents]

[2.5 hyp.yaml和opt.yaml：超参数][2.5 hyp.yaml_opt.yaml]

[2.6 P\_curve.png：准确率和置信度的关系图][2.6 P_curve.png]

[2.7 R\_curve.png ：召回率和置信度的关系图 ][2.7 R_curve.png _]

[2.8 PR\_curve.png ：精确率和召回率的关系图 ][2.8 PR_curve.png _]

[2.9 results.png ：可视化训练结果解析][2.9 results.png]

[2.10 results.txt：检测结果文本][2.10 results.txt]

[2.11 test\_batchx ：用于测试模型性能的文件夹][2.11 test_batchx]

[2.12 val\_batchx\_labels：验证集第x轮的实际标签][2.12 val_batchx_labels_x]

[🌟三、影响mAP指标的因素][mAP]

[🌟四、优化mAP的方法][mAP 1]

![](https://i-blog.csdnimg.cn/blog_migrate/e8c20818cd09c4153dfc1a21d87fee37.gif)

## 🌟一、性能指标 

### 1.1 混淆矩阵 

混淆矩阵（Confusion [Matrix][]），对分类问题预测结果的总结。使用计数值汇总正确和不正确预测的数量，并按每个类进行细分，显示了[分类模型][Link 4]进行预测时会对哪一部分产生混淆。顾名思义，真的很容易混淆╮（￣﹏￣）╭

![](https://i-blog.csdnimg.cn/blog_migrate/41a7f18d61efdeb53fb2d6b31e7c9f5d.png)

 *  T(True)：最终预测结果正确。
 *  F(False)：最后预测结果错误。
 *  P(Positive)：模型预测其是正样本(目标本身是狗，模型也预测它是狗)。
 *  N(Negative)：模型预测其是负样本(目标本身是狗，但模型预测它是个猫)。
 *  TP：样本的真实类别是正样本（P），并且模型预测的结果也是正样本（P），预测正确（T）(目标本身是狗，模型也预测它是狗，预测正确)。
 *  TN：样本的真实类别是负样本（N），并且模型将其预测成为负样本（N），预测正确（T）(目标本身不是狗，模型预测它不是狗，是个其他的东西，预测正确)。
 *  FP：样本的真实类别是负样本（N），但是模型将其预测成为正样本（P），预测错误（F）(目标本身不是狗，模型预测它是狗，预测错误)。
 *  FN：样本的真实类别是正样本（P），但是模型将其预测成为负样本（N），预测错误（F）(目标本身是狗，模型预测它不是狗，是个其他的东西，预测错误)。

### 1.2 Accuracy ：准确率（正确率） 

含义：所有预测中正确的百分比

公式：![](https://latex.csdn.net/eq?%5Ctext%20%7B%20accuracy%20%7D%3D%5Cfrac%7BT%20P+T%20N%7D%7BT%20P+F%20P+F%20N+T%20N%7D)

举个栗子：现有100只动物，分别是30只猫、50只狗和20只猪。经过模型检测之后预测正确的是20只猫、30只狗和10只猪，那么准确率（Accuracy）=（20\+30\+10）/100 = 60%。

注：通常来说正确率越高，模型越好。

### 1.3 Precision ：精确率（查准率） 

含义：指模型识别出的正确正样本数占所有被识别为正样本的样本数的比例

公式：![](https://latex.csdn.net/eq?precision%3D%5Cfrac%7BT%20P%7D%7BT%20P+F%20P%7D)

举个栗子：现有100只动物，分别是30只猫、50只狗和20只猪。经过模型检测之后结果表示有35只猫，但它认为的35只猫里面有2只狗和3只猪，所以猫预测对的只有30只，那么精确率Precision（猫）= 30/35 = 85.7%

### 1.4 Recall ：召回率（查全率） 

含义：指模型识别出的正样本数占真实正样本数的比例

公式：![](https://latex.csdn.net/eq?Recall%3D%5Cfrac%7BT%20P%7D%7BT%20P+F%20N%7D)

举个栗子：现有100只动物，分别是30只猫、50只狗和20只猪。经过模型检测之后结果表示只有25只猫，或许它把另外5只猫错认成狗和猪了吧，，那么召回率Recall（猫）= 25/30 = 83%

注：召回率越高，实际为正样本（P）被预测出来的概率越高，类似于“宁可错杀一千，绝不放过一个”。

> 在检测过程中，我们当然希望检索结果Precision越高越好，同时Recall也越高越好，但事实上这两者是矛盾的。
> 
> 举个栗子：还是看上面的动物，比如极端情况下，模型只检测出一只猫，且是正确的，那么Precision就是100%，但是Recall就很低；但是如果我们为了找全猫猫把所有结果都返回，那么Recall是100%，但是Precision当然就会很低。因此在不同的场合中需要判断希望Precision比较高或是Recall比较高。如果是做实验研究，可以绘制Precision-Recall曲线来帮助分析。

### 1.5 PR曲线 

含义：P代表的是precision（精准率），R代表的是recall（召回率），其代表的是精准率与召回率的关系，一般情况下，将recall设置为横坐标，precision设置为纵坐标。

![](https://i-blog.csdnimg.cn/blog_migrate/e744a6986b4796112bd2100d9301e51d.png)

在众多模型对数据进行学习后，如果其中一个模型的PR曲线A完全包住另一个模型B的PR曲线，则可断言A的性能优于B。

但是A和B发生交叉，那性能该如何判断呢？

我们可以根据曲线下方的面积大小来进行比较，但更常用的是平衡点F1。平衡点（BEP）是P=R时的取值（斜率为1），F1值越大，我们可以认为该学习器的性能较好。F1的计算如下所示：  
F1 = 2 \* P \* R ／( P + R )

### 1.6 AP （Average Precision）： 平均精度 

含义：PR曲线下面的面积，通常来说一个越好的模型，AP值越高

### 1.7 mAP（mean Average Precision）：均值平均精度 

含义：即各个类别AP的平均值

用于表达多类标签预测的性能，如AP一样，mAP越高，性能越好。

 *  mAP@.5
    
     *  当IoU为0.5时的mAP。
 *  mAP@.5 : .95
    
     *  当IoU为range(0.5 : 0.95 : 0.05)时的mAP的平均数。

### 1.8 F1-score与 F值(F-Measure) 

F1-score含义：是分类问题的一个衡量指标。一些多分类问题的机器学习竞赛，常常将F1-score作为最终测评的方法。它是精确率和召回率的调和平均数，最大为1，最小为0。

公式：![](https://latex.csdn.net/eq?F%201%3D%5Cfrac%7B1%7D%7B%5Cfrac%7B1%7D%7BPrecision%7D+%5Cfrac%7B1%7D%7BRecall%7D%7D%3D%5Cfrac%7B2%20%5Ctimes%20Precision%20%5Ctimes%20Recall%7D%7BPrecision+Recall%7D)

F值(F-Measure)含义：有时候我们对精确率和召回率并不是一视同仁，比如在医疗领域，我们不希望遗漏掉任何一位患者，所以应该更加重视召回率。我们用一个参数β来度量两者之间的关系，得到以下式子，是对P和R加权调和均值（不一定是平均），即F值(F-Measure)或称Fβ值：

公式：![](https://i-blog.csdnimg.cn/blog_migrate/4825f29fe1bc942b0f16edbc0c4d64a4.png)

## 🌟二、训练结果分析 

我们每次训练后，在runs/train文件夹下出现的一系列文件，如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/09da4cbee9a51e0638f8d2672d8ff5ff.png)

下面我们就逐一分析一下它们到底是干什么用的吧~

（涉及到理论知识的，大家往上滑动鼠标既可~）

### 2.1 weights ：权重 

![](https://i-blog.csdnimg.cn/blog_migrate/5d0be73a925827c39a3f2781c6fbaf9e.png)

这里存放的就是我们训练好的权重：

 *  best.pt：保存的是训练过程中在验证集上表现最好的模型权重。在训练过程中，每个epoch结束后都会对验证集进行一次评估，并记录下表现最好的模型的权重。这个文件通常用于推理和部署阶段。
 *  last.pt：保存的是最后一次训练迭代结束后的模型权重。这个文件通常用于继续训练模型，因为它包含了最后一次训练迭代结束时的模型权重，可以继续从上一次训练结束的地方继续训练模型。

使用上的区别是：当需要在之前的训练基础上继续训练时，应该使用last.pt作为起点进行训练；当需要使用训练后的模型进行推理和部署时，应该使用best.pt。

### 2.2 confusion\_matrix.png ： 混淆矩阵 

![](https://i-blog.csdnimg.cn/blog_migrate/d316db604d4cf4be76fb909e87b0514f.png)

在YOLOv5的训练结果中，confusion\_matrix.png文件是一个混淆矩阵的可视化图像，用于展示模型在不同类别上的分类效果（我这个就是一个fire的单类，所以看得不明显）。

混淆矩阵是一个n×n的矩阵，其中n为分类数目，矩阵的每一行代表一个真实类别，每一列代表一个预测类别，矩阵中的每一个元素表示真实类别为行对应的类别，而预测类别为列对应的类别的样本数。

### 2.3 F1\_curve.png ：F1曲线 

![](https://i-blog.csdnimg.cn/blog_migrate/0ca7670af9bfa35e5a5cba4986e84272.png)

F1\_curve是F1-score与置信度之间的关系：F1-score是分类问题的一个衡量指标，可以用于评估模型在检测出所有目标的情况下的精确性和完整性，是精确率precision和召回率recall的调和平均数，介于0，1之间，1是最好，0是最差。

在 YOLOv5 的训练过程中，每个训练轮次结束后，会计算出模型在验证集上的 F1-score 值，并将这些值记录下来。F1\_curve.png 就是将这些 F1-score 值绘制成的折线图，横轴表示置信度，纵轴表示 F1-score 值。通过观察 F1\_curve.png，可以了解模型在训练过程中 F1-score 值的变化情况，以及模型的训练效果。

### 2.4 events.out.tfevents：可视化文件 

events.out.tfevents：主要是保存训练阶段和评估阶段得loss值的，不需要这次训练信息的话可以删

（其实这个我也不太清楚，也没找到讲解，以上是@\-FIONASENIOR！！PEI这位大佬以及其评论区的解读，如有不对请指正！）

### 2.5 hyp.yaml和opt.yaml：超参数 

 *  hyp.yaml文件

![](https://i-blog.csdnimg.cn/blog_migrate/9aa253225e8af9133e6da533d287b734.png)

包含了训练超参数的设置，包括学习率、动量、权重衰减系数、数据增强等参数。这些超参数的设置直接影响着模型的训练效果，通过对hyp.yaml文件进行调整，可以优化模型的性能。

 *  opt.yaml文件

![](https://i-blog.csdnimg.cn/blog_migrate/87ddf32fb29b416d9a600939dd5b7942.png)

包含了train.py中间的参数的设置，如权重选择、配置文件选择、模型保存路径、训练轮次、批次大小等。这些设置会影响训练的整个流程，通过对opt.yaml文件进行调整，可以更好地控制训练过程。

在训练过程中，YOLOv5会读取hyp.yaml和opt.yaml文件中的设置，并根据这些设置进行训练。在训练完成后，这些文件也可以被用于模型的测试和部署。

### 2.6 P\_curve.png：准确率和置信度的关系图 

![](https://i-blog.csdnimg.cn/blog_migrate/d9c04e8c3830b36a298a680ff6b93cb1.png)

P\_curve.png描述随着置信度阈值的增加，P值的变化；置信度设为某一数值的时候，各个类别识别的准确率。  
在目标检测中，置信度是指检测器对于每个检测框预测目标存在的概率。在YOLOv5的训练过程中，每个检测框的置信度都会被计算出来。P\_curve.png将检测框的置信度从小到大排序，然后计算出不同置信度下的精度（Precision）值，最后将这些值绘制成一条曲线。

### 2.7 R\_curve.png ：召回率和置信度的关系图 

![](https://i-blog.csdnimg.cn/blog_migrate/dbc46858d622471db5cb87e1c19e8534.png)

R\_curve.png是用于衡量模型在不同召回率下的精确度表现。在该图表中，X轴表示召回率，范围从0到1，Y轴表示精确度，范围也是从0到1。

在目标检测中，召回率指的是模型正确检测出目标的检测框数与所有目标的数量的比值。在YOLOv7的训练过程中，每个检测框的置信度都会被计算出来。R\_curve.png将检测框的置信度从小到大排序，然后计算出不同置信度下的召回率值，最后将这些值绘制成一条曲线。

### 2.8 PR\_curve.png ：精确率和召回率的关系图 

![](https://i-blog.csdnimg.cn/blog_migrate/5f876c840bb2a166db1f73c0ac87a2ab.png)

PR\_curve.png是描述精度和召回率的关系图。精度越高，召回率越低，理想情况是(1,1)点，即在准确度很高的情况下，尽可能检测到全部的类别。二者围成的面积就是mAP值，mAP面积越接近于1，效果越好。

在目标检测中，精确率指的是模型正确检测出目标的检测框数与所有检测框数的比值，而召回率指的是模型正确检测出目标的检测框数与所有目标的数量的比值。在YOLOv5的训练过程中，每个检测框的置信度都会被计算出来。PR\_curve.png将检测框的置信度从小到大排序，然后计算出不同置信度下的精确率和召回率值，最后将这些值绘制成一条曲线。

### 2.9 results.png ：可视化训练结果解析 

![](https://i-blog.csdnimg.cn/blog_migrate/223f5a10f5f4f553739b35636003ad17.png)

 *  定位损失box\_loss：YOLO V5使用 GIOU Loss作为bounding box的损失，Box推测为GIoU损失函数均值，越小方框越准
 *  置信度损失obj\_loss：推测为目标检测loss均值，越小目标检测越准
 *  分类损失cls\_loss：推测为分类loss均值，越小分类越准
 *  val/box\_loss: 验证集bounding box损失
 *  val/obj\_loss：验证集目标检测loss均值
 *  val/cls\_loss：验证集分类loss均值，我这个项目只有fire这一类，所以为0
 *  mAP@0.5:0.95：表示在不同的IOU阈值（从0.5到0.95，步长为0.05）(0.5、0.55、0.6、0.65、0.7、0.75、0.8、0.85、0.9、0.95）上的平均mAP
 *  mAP@0.5：表示阈值大于0.5的平均mAP

一般训练结果主要观察精度和召回率波动情况（波动不是很大则训练效果较好)，然后观察mAP@0.5 & mAP@0.5:0.95 评价训练结果。

### 2.10 results.txt：检测结果文本 

![](https://i-blog.csdnimg.cn/blog_migrate/fa8342d6ed3a3d5825001d0b2c650c97.png)

results.txt是YOLOv5模型在测试集上的检测结果文本文件。  
在目标检测任务中，我们需要对测试集中的图像进行目标检测，并将检测结果输出。results.txt就是将模型在测试集上的检测结果保存在文本文件中得到的文件。对于每张测试图像，results.txt会记录检测出的目标框的位置坐标、目标类别、以及置信度分数等信息。

每行含义分别是：训练次数、GPU消耗、训练集边界框损失、训练集目标检测损失、训练集分类损失、训练集总损失、targets目标、输入图片大小、Precision、Recall、mAP@.5、mAP@.5:.95、验证集边界框损失、验证集目标检测损失、验证机分类损失

### 2.11 test\_batchx ：用于测试模型性能的文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/c585d357f0a4f22c05ef14d95725ca94.jpeg)

test\_batchx是YOLOv5模型用于保存测试集中的图像和标签信息的文件夹，其中x表示测试集的批次编号。  
在test\_batchx文件夹中，每个图像都有一个对应的标签文件，用于描述图像中目标的位置和类别信息。测试时，将test\_batchx文件夹中的图像输入到模型中进行目标检测，然后将检测结果与标签进行比较，计算模型的性能。

### 2.12 val\_batchx\_labels：验证集第x轮的实际标签 

![](https://i-blog.csdnimg.cn/blog_migrate/b62d2b94214ab9eef09ea9884b2fce5b.jpeg)  
val\_batchx\_labels 是指测试集中一个 batch 的真实标签和框的信息，其中 x 为 batch 的编号。这些信息通常包括每个样本的分类标签和相应的边界框坐标（bounding box coordinates）。  
具体来说，val\_batchx\_labels 是一个列表（list）对象，其元素个数等于 batch size。每个元素是一个元组（tuple），长度为 2。第 1 个元素是大小为 N 的 tensor，表示该 batch 中 N 个目标的分类标签；第 2 个元素也是大小为 N 的 tensor，表示该 batch 中 N 个目标的 bounding box 坐标。

## 🌟三、影响mAP指标的因素 

mAP是深度学习模型性能评估中非常重要的一种指标。比如在图像分类任务中，mAP指标可以帮助评估模型在真实图像数据上的性能，并且可以帮助选择适合任务的模型。

影响mAP指标的因素：

（1）模型的复杂度

模型的复杂度是影响mAP指标的一个重要因素。模型的复杂度越高，模型对数据集的拟合能力越强，模型在测试数据集上的性能也会越好。然而，模型的复杂度过高会导致模型训练时间过长，从而降低模型的性能。

（2）数据集的大小

数据集的大小也是影响mAP指标的一个重要因素。数据集越大，模型的性能也会越好。然而，如果数据集过大，模型的训练时间会增加，从而降低模型的性能。

（3）超参数的选择  
超参数是影响模型性能的另一个重要因素。超参数的选择会影响模型的拟合能力，从而影响模型在测试数据集上的性能。超参数的选择不当会导致模型的性能下降。

## 🌟四、优化mAP的方法 

优化mAP指标的方法：

（1）数据增强

通过对训练数据进行旋转、平移、缩放等变换，增加样本量，提高模型的鲁棒性和泛化能力。

（2）模型优化

使用更加先进的模型结构，如ResNet、Inception、EfficientNet等，或者对现有模型进行微调，如增加层数、减小学习率等。

（3）损失函数优化

选择合适的损失函数，如Focal Loss、loU Loss等，可以让模型更加关注难以识别的样本，提高模型的精度。

（4）多尺度训练

在训练过程中，使用不同的输入尺寸，可以让模型更好地适应不同大小的目标，从而提高模型的检测能力。

（5）网络融合

通过将不同的检测网络进行融合，可以提高模型的表现，如Faster R-CNN和SSD的融合。

完结~撒花✿✿ヽ(°▽°)ノ✿

> 本文参考（感谢大佬！）：
> 
> [YOLOv5训练结果性能分析\_\_tt丫的博客-CSDN博客][YOLOv5_tt_-CSDN]
> 
> [yolov7模型训练结果分析以及如何评估yolov7模型训练的效果\_把爱留给SCI的博客-CSDN博客][yolov7_yolov7_SCI_-CSDN]
