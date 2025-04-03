#### ![](https://i-blog.csdnimg.cn/blog_migrate/01f0e266fd6fb2bc7ea1237777611577.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/cf11c0f6b342f31d7902ca6930ee331b.jpeg)

## 前言 

前面简单介绍了YOLOv5的网络结构和创新点（直通车：[【YOLO系列】YOLOv5超详细解读（网络详解）][YOLO_YOLOv5]）

在接下来我们会进入到YOLOv5更深一步的学习，首先从源码解读开始。

因为我是纯小白，刚开始下载完源码时真的一脸懵，所以就先从最基础的项目目录结构开始吧~因为相关解读不是很多，所以有的是我根据作者给的英文文档自己翻译的，如有不对之处欢迎大家指正呀！这篇只是简单介绍每个文件是做什么的，大体上了解这个项目，具体的代码详解后期会慢慢更新，也欢迎大家关注我的专栏，和我一起学习呀！

源码下载地址：[mirrors / ultralytics / yolov5 · GitCode][mirrors _ ultralytics _ yolov5 _ GitCode]

![](https://i-blog.csdnimg.cn/blog_migrate/87aec300bb880db5664be59ff898a801.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/b73c94ef63ab52661adda762e17e0d7b.gif)【写论文必看】[深度学习纯小白如何从零开始写第一篇论文？看完这篇豁然开朗！][Link 1]

![](https://i-blog.csdnimg.cn/blog_migrate/b73c94ef63ab52661adda762e17e0d7b.gif)🍀本人YOLOv5源码详解系列： 

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

![](https://i-blog.csdnimg.cn/blog_migrate/9d66c6b496add6f9fddd9ce463aca2e1.gif)🌟本人YOLOv5入门实践系列：

[YOLOv5入门实践（1）——手把手带你环境配置搭建][YOLOv5_1 1]

[YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集][YOLOv5_2_labelimg]

[YOLOv5入门实践（3）——手把手教你划分自己的数据集][YOLOv5_3]

[YOLOv5入门实践（4）——手把手教你训练自己的数据集][YOLOv5_4]

[YOLOv5入门实践（5）——从零开始，手把手教你训练自己的目标检测模型（包含pyqt5界面）][YOLOv5_5_pyqt5]

![](https://i-blog.csdnimg.cn/blog_migrate/9d66c6b496add6f9fddd9ce463aca2e1.gif)🌟本人YOLOv5改进系列：

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

  


[YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等][YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]

[YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]

[YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）][YOLOv5_20_BiFormer_CVPR2023]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT]

[YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）][YOLOv5_23_MobileViTv2_ Transformer]  
  
[YOLOv5改进系列（24）——替换主干网络之MobileViTv3（移动端轻量化网络的进一步升级）][YOLOv5_24_MobileViTv3]

持续更新中。。。

## 目录 

[前言][Link 2]

[ 一、项目目录结构][Link 3]

[ 1.1 .github文件夹][1.1 .github]

[1.2 datasets][]

[ 1.3 data文件夹][1.3 data]

[ 1.4 models文件夹][1.4 models]

[1.5 runs文件夹][1.5 runs]

[1.6 utils文件夹][1.6 utils]

[1.7其他一级目录文件][1.7]

## ![](https://i-blog.csdnimg.cn/blog_migrate/b419fbcd8b4286b45458131909bb69b2.gif) 

## 一、项目目录结构 

![](https://i-blog.csdnimg.cn/blog_migrate/d48493c4b99c3cfe53fe9d3f3436d387.png)

将源码下载好并配置好环境之后，就可以看到YOLOv5的整体目录如上图所示。

接下来我们逐一分析

### 1.1 .github文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/96723319a0ad05901d276bbdf13e7137.png)

github是存放关于github上的一些“配置”的，这个不重要，我们可以不管它。

### 1.2 datasets 

![](https://i-blog.csdnimg.cn/blog_migrate/75734a20239a7e54ee4217cc63fe8d4d.png)

我们刚下载下来的源码是不包含这个文件夹的，datasets用来存放自己的数据集，分为images和labels两部分。同时每一个文件夹下，又应该分为train，val。.cache文件为缓存文件，将数据加载到内存中，方便下次调用快速。可以自命名，比如我的火焰数据集就叫“fire\_yolo\_format”。

### 1.3 data文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/38b6a61e48b1183377355d4265555065.png)

data文件夹主要是存放一些超参数的配置文件（如.yaml文件）是用来配置训练集和测试集还有验证集的路径的，其中还包括目标检测的种类数和种类的名称；还有一些官方提供测试的图片。YOLOv5 有大约 30 个超参数用于各种训练设置。更好的初始猜测会产生更好的最终结果，因此在演化之前正确初始化这些值很重要。

如果是训练自己的数据集的话，那么就需要修改其中的yaml文件。不过要注意，自己的数据集不建议放在这个路径下面，建议把数据集放到YOLOv5项目的同级目录下面。

详解：

 *  hyps文件夹\# 存放yaml格式的超参数配置文件
    
     *  hyps.scratch-high.yaml\# 数据增强高，适用于大型型号，即v3、v3-spp、v5l、v5x
     *  hyps.scratch-low.yaml \# 数据增强低，适用于较小型号，即v5n、v5s
     *  hyps.scratch-med.yaml \# 数据增强中，适用于中型型号。即v5m
 *  images \# 存放着官方给的两张测试图片
 *  scripts \# 存放数据集和权重下载shell脚本
    
     *  download\_weights.sh \# 下载权重文件，包括五种大小的P5版和P6版以及分类器版
     *  get\_coco.sh  \# 下载coco数据集
     *  get\_coco128.sh\# 下载coco128（只有128张）
 *  Argoverse.yaml \# 后面的每个.yaml文件都对应一种标准数据集格式的数据
 *  coco.yaml  \# COCO数据集配置文件
 *  coco128.yaml  \# COCO128数据集配置文件
 *  voc.yaml \# VOC数据集配置文件

### 1.4 models文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/0709de65931132a9181891074b3439a7.png)

models是模型文件夹。里面主要是一些网络构建的配置文件和函数，其中包含了该项目的四个不同的版本，分别为是s、m、l、x。从名字就可以看出，这几个版本的大小。他们的检测速度分别都是从快到慢，但是精确度分别是从低到高。如果训练自己的数据集的话，就需要修改这里面相对应的yaml文件来训练自己模型。

详解：

 *  hub \# 存放yolov5各版本目标检测网络模型配置文件
    
     *  anchors.yaml \# COCO数据的默认锚点
     *  yolov3-spp.yaml \# 带spp的yolov3
     *  yolov3-tiny.yaml \# 精简版yolov3
     *  yolov3.yaml \# yolov3
     *  yolov5-bifpn.yaml  \# 带二值fpn的yolov5l
     *  yolov5-fpn.yaml \# 带fpn的yolov5
     *  yolov5-p2.yaml \# (P2, P3, P4, P5)都输出，宽深与large版本相同，相当于比large版本能检测更小物体
     *  yolov5-p34.yaml \# 只输出(P3, P4)，宽深与small版本相同，相当于比small版本更专注于检测中小物体
     *  yolov5-p6.yaml \# (P3, P4, P5, P6)都输出，宽深与large版本相同，相当于比large版本能检测更大物体
     *  yolov5-p7.yaml \# (P3, P4, P5, P6, P7)都输出，宽深与large版本相同，相当于比large版本能检测更更大物体
     *  yolov5-panet.yaml \# 带PANet的yolov5l
     *  yolov5n6.yaml  \# (P3, P4, P5, P6)都输出，宽深与nano版本相同，相当于比nano版本能检测更大物体，anchor已预定义
     *  yolov5s6.yaml \# (P3, P4, P5, P6)都输出，宽深与small版本相同，相当于比small版本能检测更大物体，anchor已预定义
     *  yolov5m6.yaml  \# (P3, P4, P5, P6)都输出，宽深与middle版本相同，相当于比middle版本能检测更大物体，anchor已预定义
     *  yolov5l6.yaml \# (P3, P4, P5, P6)都输出，宽深与large版本相同，相当于比large版本能检测更大物体，anchor已预定义，推测是作者做实验的产物
     *  yolov5x6.yaml \# (P3, P4, P5, P6)都输出，宽深与Xlarge版本相同，相当于比Xlarge版本能检测更大物体，anchor已预定义
     *  yolov5s-ghost.yaml \# backbone的卷积换成了GhostNet形式的yolov5s，anchor已预定义
     *  yolov5s-transformer.yaml \# backbone最后的C3卷积添加了Transformer模块的yolov5s，anchor已预定义
 *  \_int\_.py \# 空的
 *  common.py  \# 放的是一些网络结构的定义通用模块，包括autopad、Conv、DWConv、TransformerLayer等
 *  experimental.py \# 实验性质的代码，包括MixConv2d、跨层权重Sum等
 *  tf.py \# tensorflow版的yolov5代码
 *  yolo.py \# yolo的特定模块，包括BaseModel，DetectionModel，ClassificationModel，parse\_model等
 *  yolov5l.yaml \# yolov5l网络模型配置文件，large版本，深度1.0，宽度1.0
 *  yolov5m.yaml \# yolov5m网络模型配置文件，middle版本，深度0.67，宽度0.75
 *  yolov5n.yaml \# yolov5n网络模型配置文件，nano版本，深度0.33，宽度0.25
 *  yolov5s.yaml \# yolov5s网络模型配置文件，small版本，深度0.33，宽度0.50
 *  yolov5x.yaml  \# yolov5x网络模型配置文件，Xlarge版本，深度1.33，宽度1.25

### 1.5 runs文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/62528d050315fb02dd50aa9ee8783b49.png)

runs是我们运行的时候的一些输出文件。每一次运行就会生成一个exp的文件夹。

![](https://i-blog.csdnimg.cn/blog_migrate/3a1c6762b7516fcb4c0d370af4509d23.png)

 详解：

 *  detect  \# 测试模型，输出图片并在图片中标注出物体和概率
 *  train \# 训练模型，输出内容，模型(最好、最新)权重、混淆矩阵、F1曲线、超参数文件、P曲线、R曲线、PR曲线、结果文件（loss值、P、R）等expn  
    
    
     *  expn \# 第n次实验数据
     *  confusion\_matrix.png\# 混淆矩阵
     *  P\_curve.png\# 准确率与置信度的关系图线
     *  R\_curve.png\# 精准率与置信度的关系图线
     *  PR\_curve.png\# 精准率与召回率的关系图线
     *  F1\_curve.png\# F1分数与置信度（x轴）之间的关系
     *  labels\_correlogram.jpg \# 预测标签长宽和位置分布
     *   results.png\# 各种loss和metrics（p、r、mAP等，详见utils/metrics）曲线
     *  results.csv\# 对应上面png的原始result数据
     *  hyp.yaml\# 超参数记录文件
     *  opt.yaml\# 模型可选项记录文件
     *  train\_batchx.jpg\# 训练集图像x（带标注）
     *  val\_batchx\_labels.jpg\# 验证集图像x（带标注）
     *  val\_batchx\_pred.jpg\# 验证集图像x（带预测标注）
     *  weights\# 权重
     *  best.pt\# 历史最好权重
     *  last.pt\# 上次检测点权重
     *  labels.jpg\# 4张图， 4张图，（1，1）表示每个类别的数据量

 （1，2）真实标注的 bounding\_box

 （2，1） 真实标注的中心点坐标

 （2，2）真实标注的矩阵宽高

### 1.6utils文件夹 

![](https://i-blog.csdnimg.cn/blog_migrate/8f1ba9ad0ba5720befdba2d2cbfb8c68.png)

 utils工具文件夹。存放的是工具类的函数，里面有loss函数，metrics函数，plots函数等等。

 详解：

 *  aws\# 恢复中断训练,和aws平台使用相关的工具
 *  flask\_rest\_api\# 和flask 相关的工具
 *  google\_app\_engine\# 和谷歌app引擎相关的工具
 *  loggers\# 日志打印
 *  \_init\_.py\# notebook的初始化，检查系统软件和硬件
 *  activations.py\# 激活函数
 *  augmentations\# 存放各种图像增强技术
 *  autoanchor.py\# 自动生成锚框
 *  autobatch.py\# 自动生成批量大小
 *  benchmarks.py \# 对模型进行性能评估（推理速度和内存占用上的评估）
 *  callbacks.py\# 回调函数，主要为logger服务
 *  datasets\# dateset和dateloader定义代码
 *  downloads.py\# 谷歌云盘内容下载
 *  general.py\# 全项目通用代码，相关实用函数实现
 *  loss.py\# 存放各种损失函数
 *  metrics.py\# 模型验证指标，包括ap，混淆矩阵等
 *  plots.py\# 绘图相关函数，如绘制loss、ac曲线，还能单独将一个bbox存储为图像
 *  torch\_utils.py \# 辅助函数

### 1.7其他一级目录文件 

![](https://i-blog.csdnimg.cn/blog_migrate/652e120c4625382be7f8e3cfd357c9cd.png)

 详解：

 *  .dockerignore \# docker的ignore文件
 *  .gitattributes \# 用于将.[ipynb][]后缀的文件剔除GitHub语言统计
 *  .gitignore\# docker的ignore文件
 *  CONTRIBUTING.md \# markdown格式说明文档
 *  detect.py \# 目标检测预测脚本
 *  export.py \# 模型导出
 *  hubconf.py \# pytorch hub相关
 *  LICENSE  \# 证书
 *  README.md  \# markdown格式说明文档
 *  requirements.txt  \# 可以通过pip install requirement进行依赖环境下载
 *  setup.cfg \# 项目打包文件
 *  train.py \# 目标检测训练脚本
 *  tutorial.ipynb \# 目标检测上手教程
 *  val.py \# 目标检测验证脚本
 *  yolov5s.pt \# coco数据集模型预训练权重，运行代码的时候会自动从网上下载

> 本文参考：
> 
> [YOLOV5学习笔记（四）——项目目录及代码讲解][YOLOV5]
> 
> [YOLOv5-6.2版本代码Project逐文件详解][YOLOv5-6.2_Project]

![](https://i-blog.csdnimg.cn/blog_migrate/d335a23a27bdd33331d979e2d58df018.gif)


[YOLO_YOLOv5]: https://blog.csdn.net/weixin_43334693/article/details/129312409?spm=1001.2014.3001.5501
[mirrors _ ultralytics _ yolov5 _ GitCode]: https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator
[Link 1]: https://blog.csdn.net/weixin_43334693/article/details/133617849?spm=1001.2014.3001.5501
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764
[YOLOv5_1 1]: https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501
[YOLOv5_2_labelimg]: https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501
[YOLOv5_3]: https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501
[YOLOv5_4]: https://blog.csdn.net/weixin_43334693/article/details/130043351?spm=1001.2014.3001.5501
[YOLOv5_5_pyqt5]: https://blog.csdn.net/weixin_43334693/article/details/130044342?spm=1001.2014.3001.5501
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
[YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]: https://blog.csdn.net/weixin_43334693/article/details/131513850?spm=1001.2014.3001.5501
[YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]: https://blog.csdn.net/weixin_43334693/article/details/131552028?spm=1001.2014.3001.5501
[YOLOv5_15]: https://blog.csdn.net/weixin_43334693/article/details/131613721?spm=1001.2014.3001.5501
[YOLOv5_16_EMA_ICASSP2023]: https://blog.csdn.net/weixin_43334693/article/details/131973273?spm=1001.2014.3001.5501
[YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]: https://blog.csdn.net/weixin_43334693/article/details/131999141?spm=1001.2014.3001.5501
[YOLOv5_18_Neck_AFPN_PAFPN]: https://blog.csdn.net/weixin_43334693/article/details/132070079?spm=1001.2014.3001.5501
[YOLOv5_19_Swin TransformerV1_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132161488?spm=1001.2014.3001.5501
[YOLOv5_20_BiFormer_CVPR2023]: https://blog.csdn.net/weixin_43334693/article/details/132203200?spm=1001.2014.3001.5501
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502
[YOLOv5_24_MobileViTv3]: https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502
[Link 2]: #%E5%89%8D%E8%A8%80
[Link 3]: #%C2%A0%E4%B8%80%E3%80%81%E9%A1%B9%E7%9B%AE%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84
[1.1 .github]: #%C2%A01.1%C2%A0.github%E6%96%87%E4%BB%B6%E5%A4%B9
[1.2 datasets]: #1.2%C2%A0datasets
[1.3 data]: #%C2%A01.3%C2%A0data%E6%96%87%E4%BB%B6%E5%A4%B9
[1.4 models]: #%C2%A01.4%C2%A0models%E6%96%87%E4%BB%B6%E5%A4%B9
[1.5 runs]: #1.5%C2%A0runs%E6%96%87%E4%BB%B6%E5%A4%B9
[1.6 utils]: #1.6%C2%A0utils%E6%96%87%E4%BB%B6%E5%A4%B9
[1.7]: #1.7%E5%85%B6%E4%BB%96%E4%B8%80%E7%BA%A7%E7%9B%AE%E5%BD%95%E6%96%87%E4%BB%B6
[ipynb]: https://so.csdn.net/so/search?q=ipynb&spm=1001.2101.3001.7020
[YOLOV5]: https://blog.csdn.net/HUASHUDEYANJING/article/details/126086708
[YOLOv5-6.2_Project]: https://blog.csdn.net/qq_53627591/article/details/128555629