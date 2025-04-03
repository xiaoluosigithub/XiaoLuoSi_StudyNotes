![](https://i-blog.csdnimg.cn/blog_migrate/a735e2917bf8f53de525c55be5a0a512.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/cba5712f3cc13abdcc15ac751fae103a.png)

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

[YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]

[YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

![](https://i-blog.csdnimg.cn/blog_migrate/3d0a0a91e7a91cef9cbf6b7245bd6108.gif)

目录

[🚀一、MPDIoU介绍 ][MPDIoU_]

[1.1 简介][1.1]

[1.2 最小点距离交并比][1.2]

[1.3 MPDIoU的损失函数][1.3 MPDIoU]

[1.4 实验][1.4]

[🚀二、更换MPDIoU方法][MPDIoU]

[第①步 配置metric.py文件][_metric.py]

[第②步 配置loss.py文件][_loss.py]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/9a4331f1f372c604cb4d5ef25e8cb26d.gif)

## 🚀一、MPDIoU介绍 

>  *  论文题目：《MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression》
>  *  原文地址：[arxiv.org/pdf/2307.07662v1.pdf][arxiv.org_pdf_2307.07662v1.pdf]

### ![](https://i-blog.csdnimg.cn/blog_migrate/aef262906d69374479c71e6f64956f9c.png)1.1 简介 

在目标检测任务中，损失函数经常用于度量神经网络预测信息与期望标签的距离，预测信息越接近期望信息，损失函数值越小。在YOLO系列使用CIoU，CIoU 反映长宽比的不同，而不是长宽位置与置信度之间存在的实际差距，这就会在检测过程中产生一定的问题。

如图8（a）所示，图中外框为实际检测框，内框为预测框，当中心点重合长宽比一致时，CIoU就会失效。另外如图8（b）所示，在多个预测框大面积重叠的情况下，不能反映出实际情况。

![](https://i-blog.csdnimg.cn/blog_migrate/53f389e29431e0021115d2a495727934.png)

为了解决上述问题，作者充分探索了水平矩形的几何特征，提出了一种基于最小点距离的边界框相似度比较度量——MPDIoU，其中包含了现有损失函数中考虑的所有相关因素，例如重叠或非重叠面积、中心点距离以及宽度和高度的偏差，同时简化了计算过程。在此基础上，作者提出了一种基于MPDIoU的边界框回归损失函数，称为MLPDIoU。

实验结果表明，将MPDIoU损失函数应用于最先进的实例分割（如YOLACT）和目标检测（如YOLOv7）模型，在PASCAL VOC、MS COCO和IIIT5k数据集上优于现有的损失函数。

### 1.2 最小点距离交并比 

作者设计了一种新颖的基于交并比的度量标准，名为MPDIoU，直接最小化预测边界框与实际标注边界框之间的左上角和右下角点距离。

该方法直接最小化预测边界框与实际标注边界框之间的左上角和右下角点距离。在训练阶段，通过公式迫使模型预测的每个边界框![B^{prd}=[x^{prd},y^{prd},w^{prd},h^{prd}]^{T}](https://latex.csdn.net/eq?B%5E%7Bprd%7D%3D%5Bx%5E%7Bprd%7D%2Cy%5E%7Bprd%7D%2Cw%5E%7Bprd%7D%2Ch%5E%7Bprd%7D%5D%5E%7BT%7D)接近其实际边界框![B_{gt}=[x_{gt},y_{gt},w_{gt},h_{gt}]^{T}](https://latex.csdn.net/eq?B_%7Bgt%7D%3D%5Bx_%7Bgt%7D%2Cy_%7Bgt%7D%2Cw_%7Bgt%7D%2Ch_%7Bgt%7D%5D%5E%7BT%7D)：

![](https://i-blog.csdnimg.cn/blog_migrate/641ecb2fef9310f5e7bf83710ac34198.png)

其中，![B_{gt}](https://latex.csdn.net/eq?B_%7Bgt%7D)是实际边界框的集合，![\Theta](https://latex.csdn.net/eq?%5CTheta)是用于回归的深度模型的参数。![L](https://latex.csdn.net/eq?L)的典型形式是![L_{n}-norm](https://latex.csdn.net/eq?L_%7Bn%7D-norm)

MPDIoU的计算过程总结在算法1中：

![27c14de8a52c96f1e096d04934f81185.png](https://i-blog.csdnimg.cn/blog_migrate/bf211fe9f252b5f723948fe4415fd697.png)

总结一下：

1.  提出的MPDIoU简化了两个边界框之间的相似性比较
2.  适用于重叠或非重叠的边界框回归

因此，在2D/3D计算机视觉任务中，MPDIoU可以很好地替代交并比作为所有性能指标的度量。

### 1.3 MPDIoU的损失函数 

MPDIoU的损失函数公式如下：

![](https://i-blog.csdnimg.cn/blog_migrate/0dd77b8472917d1b52ec8b5ef3d55d0b.png)

因此，现有边界框回归损失函数的所有因素都可以通过4个点的坐标来确定。

转换公式如下所示：

![c9457256d91005c2e0a706fbb3a4aac3.png](https://i-blog.csdnimg.cn/blog_migrate/5687e92940e6d0059e51b159d60eced9.png)

（公式解读大家自己看原文或是别的大佬解读吧，数学白痴先行告退...orz）

### 1.4 实验 

（1）目标检测的实验

![](https://i-blog.csdnimg.cn/blog_migrate/b87e5f3ea390ac5c1922a9f96f64fceb.png)

（2）字符级场景文本定位的实验

![](https://i-blog.csdnimg.cn/blog_migrate/57048f11695f52859a14b9da90e55c85.png)

（3）实例分割的实验结果

![](https://i-blog.csdnimg.cn/blog_migrate/14e87e05626bc9d61947944f1f821694.png)

## 🚀二、更换MPDIoU方法 

#### 第①步 配置metric.py文件 

首先找到 utils/metrics.py 文件夹下的 bbox\_iou 函数，然后将函数整个替换成下面的代码：

```java
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIoU=False, WIoU=False, Focal=False,
              MPDIoU=False, alpha=1, gamma=0.5, scale=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        self = WIoU_Scale(1 - (inter / union))

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU or MPDIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU or MPDIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                        b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter / (union + eps),
                                                                                                 gamma)  # Focal_CIoU
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps),
                                                                                      gamma)  # Focal_EIou
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif MPDIoU:
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                d12 = ((b2_x1 - b1_x1) - (b2_y1 - b1_y1)) ** 2
                d22 = ((b2_x2 - b1_x2) - (b2_y2 - b1_y2)) ** 2
                return iou - ((d12+d22)/(cw2+ ch2))
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                        inter / (union + eps), gamma)  # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            elif WIoU:
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp(
                        (rho2 / c2)), iou  # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2))  # WIoU v1
            if Focal:
                return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)  # Focal_DIoU
            else:
                return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps),
                                                                                      gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter / (union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU
class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''

    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1
```

#### 第②步 配置loss.py文件 

然后再找到 utils/loss.py文件夹下的 \_\_call\_\_函数，把Regression loss中计算IoU的代码，换成下面这句：

```java
iou = bbox_iou(pbox, tbox[i], MPDIoU=True, scale=True)
                if type(iou) is tuple:
                    if len(iou) == 2:
                        lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
                        iou = iou[0].squeeze()
                    else:
                        lbox += (iou[0] * iou[1]).mean()
                        iou = iou[2].squeeze()
                else:
                    lbox += (1.0 - iou.squeeze()).mean()  # iou loss
                    iou = iou.squeeze()
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/db4b528cab9523470f915fd0322d6d5a.png)

![](https://i-blog.csdnimg.cn/blog_migrate/aa438630a4186ff2e7c4d8bb6edae98c.png)

就酱~

> 代码参考：
> 
> [损失函数篇 | YOLOv8 更换损失函数之 MPDIoU | 《2023 一种用于高效准确的边界框回归的损失函数》\_迪菲赫尔曼的博客-CSDN博客][_ YOLOv8 _ MPDIoU _ _2023 _-CSDN]

PS：

在我的数据集上，MPDIoU比WIoU涨了0.9，效果还是不错的~

WIoU：

![](https://i-blog.csdnimg.cn/blog_migrate/691026f1a960fa1d2f68d428100bc40c.png)

MPDIoU：

![](https://i-blog.csdnimg.cn/blog_migrate/8cb13aeb4cbe8746a6087e677a12c767.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/2fdc8395aad28a713580831d1cfa0f0b.gif)


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
[YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]: https://blog.csdn.net/weixin_43334693/article/details/131552028?spm=1001.2014.3001.5501
[YOLOv5_15]: https://blog.csdn.net/weixin_43334693/article/details/131613721?spm=1001.2014.3001.5502
[YOLOv5_16_EMA_ICASSP2023]: https://blog.csdn.net/weixin_43334693/article/details/131973273?spm=1001.2014.3001.5501
[MPDIoU_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81MPDIoU%E4%BB%8B%E7%BB%8D%C2%A0
[1.1]: #%E2%80%8B%E7%BC%96%E8%BE%911.1%20%E7%AE%80%E4%BB%8B
[1.2]: #1.2%20%E6%9C%80%E5%B0%8F%E7%82%B9%E8%B7%9D%E7%A6%BB%E4%BA%A4%E5%B9%B6%E6%AF%94
[1.3 MPDIoU]: #1.3%20MPDIoU%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0
[1.4]: #1.4%20%E5%AE%9E%E9%AA%8C
[MPDIoU]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%9C%A8backbone%E6%9C%AB%E7%AB%AF%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[_metric.py]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%C2%A0%E9%85%8D%E7%BD%AEmetric.py%E6%96%87%E4%BB%B6
[_loss.py]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%C2%A0%E9%85%8D%E7%BD%AEloss.py%E6%96%87%E4%BB%B6
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[arxiv.org_pdf_2307.07662v1.pdf]: https://arxiv.org/pdf/2307.07662v1.pdf
[_ YOLOv8 _ MPDIoU _ _2023 _-CSDN]: https://yolov5.blog.csdn.net/article/details/132002708?spm=1001.2014.3001.5502
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