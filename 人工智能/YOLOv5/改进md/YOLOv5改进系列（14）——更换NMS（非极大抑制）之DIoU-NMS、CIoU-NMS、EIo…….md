![](https://i-blog.csdnimg.cn/blog_migrate/15f2cc596fe33b25ec92ec7b5f79832d.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/9ea39c12e113aa10c6ea02402e7d3210.png)

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

[YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等][YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]

![](https://i-blog.csdnimg.cn/blog_migrate/6e7df20d56b4203546bb53ba6b10bb0e.gif)

目录

[☀️一、NMS（非极大抑制）简介][NMS]

[1.1 什么是NMS？][1.1 _NMS]

[1.2 NMS的计算过程][1.2 NMS]

[1.3 NMS的局限性][1.3 NMS]

[1.4 NMS的改进思路][1.4 NMS]

[☀️二、DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、 SIoU-NMS][DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _ SIoU-NMS]

[🌲2.1 更换DIoU-NMS][2.1 _DIoU-NMS]

[第①步 修改general.py][_general.py]

[第②步 更换NMS][_NMS]

[🌲2.2 更换其他的NMS][2.2 _NMS]

[☀️三、Soft-NMS][Soft-NMS]

[第①步 修改general.py][_general.py]

[第②步 更换NMS][_NMS]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/dab4cd356232f776767a88a126b2bf06.gif)

## ☀️一、NMS（非极大抑制）简介 

### 1.1 什么是NMS？ 

NMS（non maximum suppression）即非极大抑制，顾名思义就是抑制不是极大值的元素，搜索局部的极大值。在最近几年常见的物体检测算法（包括RCNN、SPPNet、Fast-RCNN、Faster-RCNN等）中，最终都会从一张图片中找出很多个可能是物体的矩形框，然后为每个矩形框为做类别分类概率。

如果用一句话概括NMS的意思就是：筛选出一定区域内属于同一种类别得分最大的框。

如下图，网络模型可以给每个检测框一个score，score越大，说明检测框越接近真实值。

![](https://i-blog.csdnimg.cn/blog_migrate/aed9d82ceee2097a1fc1c6d3eeee2c9e.png) 《老友记》yyds！

然后非最大值抑制的目的是删除score小的框，只剩下sore最大作为最终的预测结果。

![](https://i-blog.csdnimg.cn/blog_migrate/c61972d5e21807a1230cc13fb179dca0.png)

### 1.2 NMS的计算过程 

 *  定义置信度阈值和IOU阈值取值
 *  按置信度降序排列边界框bounding\_box
 *  从bbox\_list中删除置信度小于阈值的预测框
 *  循环遍历剩余框，首先挑选置信度最高的框作为候选框
 *  接着计算其他和候选框属于同一类的所有预测框和当前候选框的IOU
 *  如果上述任两个框的IOU的值大于IOU阈值，那么从box\_list中移除置信度较低的预测框
 *  重复此操作，直到遍历完列表中的所有预测框

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6007aac6c76014ab34a281a9200f2065.png)

### 1.3 NMS的局限性 

（1）需要手动设置阈值，阈值的设置会直接影响重叠目标的检测，太大造成误检，太小达不到理想情况。

（2）低于阈值的直接设置score为0，做法就比较麻烦了

（3）通过IoU来评估，IoU的做法对目标框尺度和距离的影响不同

### 1.4 NMS的改进思路 

（1）根据手动设置阈值的缺陷，通过自适应的方法在目标稀疏时使用小阈值，目标稠密时使用大阈值。例如Adaptive NMS  
（2）由于将低于阈值的直接置为0的做法比较困难，所以我们可以通过将其根据IoU大小来进行惩罚衰减，则变得更加平滑。例如Soft NMS，Softer NMS  
（3）IoU的做法存在一定缺陷，改进思路是将目标尺度、距离引进IoU的考虑中。如DIoU等

## ☀️二、DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、 SIoU-NMS 

### 🌲2.1 更换DIoU-NMS 

> 论文：[https://arxiv.org/pdf/1911.08287.pdf][https_arxiv.org_pdf_1911.08287.pdf].

一个成熟的IoU衡量指标应该要考虑预测框与真实框的重叠面积、中心点距离、长宽比三个方面。但是IoU 只考虑到了预测框与真实框重叠区域，并没有考虑到中心点距离、长宽比。

DIoU-NMS在DIoUloss一文中提出，不仅仅考虑IoU，还考虑两个框中心点之间的距离。如果两个框之间IoU比较大，但是两个框的中心距离比较大时，可能会认为这是两个物体的框而不会被过滤掉。由于DIoU的计算考虑到了两框中心点位置的信息，故使用DIoU进行评判的nms效果更符合实际，效果更优。

公式：![](https://i-blog.csdnimg.cn/blog_migrate/f30bc25ef98875888774d4dca7e9c07d.png)

#### 第①步 修改general.py 

在YOLOv5当中，作者是直接调用了Pytorch官方的NMS的API。

也就是在general.py中的non\_max\_suppression函数中

```java

```

首先将下面一段代码粘贴到utils/general.py，重新定义NMS模块。这里的计算IoU的函数bbox\_iou则是直接引用了YOLOv5中的代码，其简洁的集成了CIoU、SIoU、EIoU、GIoU、DIoU 的计算。

```java
def NMS(boxes, scores, iou_thres, class_nms='CIoU'):
    # class_nms=class_nms
    GIoU=CIoU=DIoU=EIoU=SIoU=False
    if class_nms == 'CIoU':
        CIoU=True
    elif class_nms == 'DIoU':
        DIoU=True
    elif class_nms == 'GIoU':
        GIoU=True
    elif class_nms == 'EIoU':
        EIoU=True
    else :
        SIoU=True
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        index = B[0]
        keep.append(index)
        if B.numel() == 1: break
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, EIoU=EIoU, SIoU=SIoU)
        inds = torch.nonzero(iou <= iou_thres).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep)
```

#### 第②步 更换NMS 

然后我们将non\_max\_suppression 函数中的

```java
i = torchvision.ops.nms(boxes, scores, iou_thres)
```

替换为

```java
i = NMS(boxes, scores, iou_thres, class_nms='DIoU')
```

这样就可以还是训练了~

### 🌲2.2 更换其他的NMS 

其余几个方法都是一样的，只需要在第②步改个名称即可：

 *  DloU：

```java
i = NMS(boxes, scores, iou_thres, class_nms='DIoU')
```

 *  SloU：

```java
i = NMS(boxes, scores, iou_thres, class_nms='SIoU')
```

 *  GloU：

```java
i = NMS(boxes, scores, iou_thres, class_nms='GIoU')
```

 *  EloU：

```java
i = NMS(boxes, scores, iou_thres, class_nms='EIoU')
```

## ☀️三、Soft-NMS 

> 论文：[https://arxiv.org/abs/1704.04503][https_arxiv.org_abs_1704.04503]

根据前面对目标检测中NMS的算法描述，易得出传统NMS的不足：如果一个物体在另一个物体重叠区域出现，即当两个目标框接近时，分数更低的框就会因为与之重叠面积过大而被删掉，从而导致对该物体的检测失败并降低了算法的平均检测率。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ce6c82732aff38c5a177af95097e8509.png)

上图中，有两匹马，都是待检测目标，也有两个检测到的框，得分分别是0.80和0.95  
如果用NMS算法，得分最高的框是红色框，得分0.95，而绿色框与红色框通过计算IoU，肯定是大于一般我们设置的0.5的，那么绿色框就会被删除，导致少检测到一匹马的情况。此外，NMS算法设置阈值也比较麻烦，如果设置过小，那么会出先这样的事情，少检测到目标；如果设置过大，又会经常出先误检。

因此，出现升级版Soft-NMS。具体流程就是我们把NMS算法中去除其他边界框改成，修改其他边界框的置信度。

#### 第①步 修改general.py 

同样，首先将下面一段代码粘贴到utils/general.py，重新定义NMS模块。

```java
def my_soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
 
    bboxes = bboxes.contiguous()
 
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # 计算每个box的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 首先对所有得分进行一次降序排列,仅此一次,以提高后续查找最大值速度. oeder为降序排列后的索引
    _, order = scores.sort(0, descending=True)
    # NMS后,保存留下来的边框
    keep = []
 
    while order.numel() > 0:
        if order.numel() == 1:  # 仅剩最后一个box的索引
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留首个得分最大的边框box索引,i为scores中实际坐标
            keep.append(i)
        # 巧妙使用tersor.clamp()函数求取order中当前框[0]之外每一个边框,与当前框[0]的最大值和最小值
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # 求取order中其他每一个边框与当前边框的交集面积
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        # 计算order中其他每一个框与当前框的IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # 共order.numel()-1个
 
        idx = (iou > iou_thresh).nonzero().squeeze()  # 获取order中IoU大于阈值的其他边框的索引
        if idx.numel() > 0:
            iou = iou[idx]
            newScores = torch.exp(-torch.pow(iou, 2) / sigma)  # 计算边框的得分衰减
            scores[order[idx + 1]] *= newScores  # 更新那些IoU大于阈值的边框的得分
 
        newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
        if newOrder.numel() == 0:
            break
        else:
            newScores = scores[order[newOrder + 1]]
            maxScoreIndex = torch.argmax(newScores)
 
            if maxScoreIndex != 0:
                newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
            # 更新order.
            order = order[newOrder + 1]
 
    # 返回保留下来的所有边框的索引值,类型torch.LongTensor
    return torch.LongTensor(keep)
```

#### 第②步 更换NMS 

将general.py中将NMS改为soft nms。

这步也是和上面一样，将non\_max\_suppression 函数中的

```java
i = torchvision.ops.nms(boxes, scores, iou_thres)
```

替换为

```java
i = my_soft_nms(boxes, scores, iou_thres)  #
```

最后就可以开始训练了~

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

![](https://i-blog.csdnimg.cn/blog_migrate/64923ef5714873f1b50af2ac7205bdde.gif)


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
[YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC]: https://blog.csdn.net/weixin_43334693/article/details/131513850?spm=1001.2014.3001.5502
[NMS]: #%E2%98%80%EF%B8%8F%E4%B8%80%E3%80%81NMS%EF%BC%88%E9%9D%9E%E6%9E%81%E5%A4%A7%E6%8A%91%E5%88%B6%EF%BC%89%E7%AE%80%E4%BB%8B
[1.1 _NMS]: #1.1%E4%BB%80%E4%B9%88%E6%98%AFNMS%EF%BC%9F
[1.2 NMS]: #1.2%20NMS%E7%9A%84%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B
[1.3 NMS]: #1.3%C2%A0%C2%A0NMS%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7
[1.4 NMS]: #1.4%20NMS%E7%9A%84%E6%94%B9%E8%BF%9B%E6%80%9D%E8%B7%AF
[DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _ SIoU-NMS]: #%F0%9F%8C%B2%E4%BA%8C%E3%80%81%E4%B8%80%E4%BA%9B%E5%B8%B8%E8%A7%81%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0
[2.1 _DIoU-NMS]: #%F0%9F%8C%B22.1%20%E6%9B%B4%E6%8D%A2DIoU-NMS
[_general.py]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%20%E4%BF%AE%E6%94%B9general.py
[_NMS]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%20%E6%9B%B4%E6%8D%A2NMS
[2.2 _NMS]: #%F0%9F%8C%B22.2%20%E6%9B%B4%E6%8D%A2%E5%85%B6%E4%BB%96%E7%9A%84NMS
[Soft-NMS]: #%E2%98%80%EF%B8%8F%E4%B8%89%E3%80%81Soft-NMS
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_pdf_1911.08287.pdf]: https://arxiv.org/pdf/1911.08287.pdf
[https_arxiv.org_abs_1704.04503]: https://arxiv.org/abs/1704.04503
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