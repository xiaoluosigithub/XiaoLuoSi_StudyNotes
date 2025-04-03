![](https://i-blog.csdnimg.cn/blog_migrate/7c409eb6c25f818abbf65d2a7e0dda4f.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/c296605ef2b5f6278e36753757763eef.png)

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

![](https://i-blog.csdnimg.cn/blog_migrate/08366df22961291ce29c003c99086da4.gif)

目录

[☀️一、不同IoU的介绍][IoU]

[☀️二、IoU、GIoU、DIoU、CIoU][IoU_GIoU_DIoU_CIoU]

[☀️三、EIoU][EIoU]

[3.1 简介][3.1]

[3.2 添加步骤][3.2]

[第①步 配置metric.py文件][_metric.py]

[第②步 配置loss.py文件][_loss.py]

[☀️四、Alpha IoU][Alpha IoU]

[4.1 简介][4.1]

[4.2 添加步骤][4.2]

[第①步 配置metric.py文件][_metric.py]

[第②步 配置loss.py文件][_loss.py]

[☀️五、SIoU][SIoU]

[5.1 简介][5.1]

[5.2 添加步骤][5.2]

[第①步 配置metric.py文件][_metric.py]

[第②步 配置loss.py文件][_loss.py]

[☀️六、WIoU][WIoU]

[6.1 简介][6.1]

[6.2 添加步骤][6.2]

[第①步 配置metric.py文件][_metric.py]

[第②步 配置loss.py文件][_loss.py]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/3c67b1a83740fb1fa5860794ee867b7c.gif)

## ☀️一、不同IoU的介绍 

要学会修改代码前一定要先了解这几个IoU的区别和计算方法：

 *  IoU Loss：主要考虑检测框和目标框重叠面积。
 *  GIoU Loss：在IoU的基础上，解决边界框不相交时loss等于0的问题。
 *  DIoU Loss：在IoU和GIoU的基础上，考虑边界框中心点距离的信息。
 *  CIoU Loss：在DIoU的基础上，考虑边界框宽高比的尺度信息。
 *  EIoU Loss：在CIoU的基础上，解决了纵横比的模糊定义，并添加Focal Loss解决BBox回归中的样本不平衡问题。
 *  αIoU Loss：通过调节α，使探测器更灵活地实现不同水平的bbox回归精度。
 *  SIoU Loss：在EIoU的基础上，加入了类别信息的权重因子，以提高检测模型的分类准确率。
 *  WIoU Loss：解决质量较好和质量较差的样本间的BBR平衡问题

更多详细的内容在上一篇就已经详细介绍过了，快上直通车：

> [损失函数：IoU、GIoU、DIoU、CIoU、EIoU、alpha IoU、SIoU、WIoU超详细精讲及Pytorch实现][IoU_GIoU_DIoU_CIoU_EIoU_alpha IoU_SIoU_WIoU_Pytorch]

## ☀️二、IoU、GIoU、DIoU、CIoU 

这几个IoU在YOLO v5的源码中都有提供，在utils/metrics.py文件夹下的，bbox\_iou函数里。

![](https://i-blog.csdnimg.cn/blog_migrate/4a690e00ade56add8a4cd4baf1fdfecc.png)

这个函数是用来计算矩阵框间的IoU的

```java
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
```

通过这行代码我们可以看出，GIoU，DIoU，CIoU的bool值如果全部为False时，会返回最普通的IoU，如果其中一个为True的时候，即返回设定为True的那个IoU

通常用来在utils/loss.py文件夹下的\_\_call\_\_函数中计算回归损失（bbox损失）

![](https://i-blog.csdnimg.cn/blog_migrate/8b46f3252c49527fe27c2ff096230967.png)

可以看出YOLOv5中bbox\_iou其默认用的是CIoU

## ☀️三、EIoU 

### 3.1 简介 

EIoU 是在 CIoU 的惩罚项基础上将预测框和真实框的纵横比的影响因子拆开，分别计算预测框和真实框的长和宽，来解决 CIoU 存在的问题。

EIoU包括三个部分：IoU损失、距离损失、高宽损失（重叠面积、中心点举例、高宽比）。高宽损失直接最小化了预测目标边界框和真实边界框的高度和宽度的差异，使其有更快的收敛速度和更好的定位结果。

![](https://i-blog.csdnimg.cn/blog_migrate/5b04adc729888f3cf51eac5bd6e7b1ca.png#pic_center)

### 3.2 添加步骤 

#### 第①步 配置metric.py文件 

首先我们找到刚才的utils/metrics.py文件夹下的bbox\_iou函数，然后将函数整个替换成下面的代码：

```java
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False,  EIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
```

#### 第②步 配置loss.py文件 

然后再找到utils/loss.py文件夹下的\_\_call\_\_函数，把Regression loss中计算IoU的代码，换成下面这句：

```java
iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False, EIoU=True)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/db4b528cab9523470f915fd0322d6d5a.png)

![](https://i-blog.csdnimg.cn/blog_migrate/1f67abc2c21b0decbabef252fa940244.png)

## ☀️四、Alpha IoU 

### 4.1 简介 

作者将现有的基于IoU Loss推广到一个新的Power IoU系列 Loss，该系列具有一个Power IoU项和一个附加的Power正则项，具有单个Power参数α，称这种新的损失系列为α-IoU Loss。

通过调节α，使检测器在实现不同水平的bbox回归精度方面具有更大的灵活性。并且α-IoU 对小数据集和噪声的鲁棒性更强。

通过实验发现，在大多数情况下，取α=3 的效果最好。

### 4.2 添加步骤 

#### 第①步 配置metric.py文件 

首先我们找到刚才的utils/metrics.py文件夹下的bbox\_iou函数，然后将函数整个替换成下面的代码，因为上文也提到α=3时效果最好，所以这里我们将alpha的值设置为3：

```java
def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=3, eps=1e-7):
    # Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)
    # beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union) # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou
```

#### 第②步 配置loss.py文件 

然后再找到utils/loss.py文件夹下的\_\_call\_\_函数，将bbox\_iou换成bbox\_alpha\_iou

```java
iou = bbox_alpha_iou(pbox.T, tbox[i], x1y1x2y2=False, alpha=3, CIoU=True)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/db4b528cab9523470f915fd0322d6d5a.png)

![](https://i-blog.csdnimg.cn/blog_migrate/52c165ec8bd846cfbeeca2c70c07632b.png)

## ☀️五、SIoU 

### 5.1 简介 

SIoU考虑到期望回归之间向量的角度，重新定义角度惩罚度量，它可以使预测框快速漂移到最近的轴，随后则只需要回归一个坐标（X或Y），这有效地减少了自由度的总数。应用于传统的神经网络和数据集，表明SIoU提高了训练的速度和推理的准确性。

SIoU进一步考虑了真实框和预测框之间的向量角度，重新定义相关损失函数，具体包含四个部分：角度损失(Angle cost)、距离损失(Distance cost)、形状损失(Shape cost)、IoU损失（IoU cost）。

### 5.2 添加步骤 

#### 第①步 配置metric.py文件 

首先我们找到刚才的utils/metrics.py文件夹下的bbox\_iou函数，然后将函数整个替换成下面的代码：

```java
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
 
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
 
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
    if GIoU or DIoU or CIoU or SIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if SIoU:    # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            # angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - np.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            return iou - 0.5 * (distance_cost + shape_cost)
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
```

#### 第②步 配置loss.py文件 

然后再找到utils/loss.py文件夹下的\_\_call\_\_函数，把Regression loss中计算IoU的代码，换成下面这句：

```java
iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, SIoU=True)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/db4b528cab9523470f915fd0322d6d5a.png)

![](https://i-blog.csdnimg.cn/blog_migrate/9e335be98ba73ae8991b06e20676521a.png)

## ☀️六、WIoU 

### 6.1 简介 

传统的Intersection over Union（IoU）只考虑了预测框和真实框的重叠部分，没有考虑两者之间的区域，导致在评估结果时可能存在偏差。基于这一思想，作者提出了一种基于IoU的损失，该损失具有动态非单调FM，名为Wise IoU（WIoU）。

### 6.2 添加步骤 

#### 第①步 配置metric.py文件 

首先我们找到刚才的utils/metrics.py文件夹下的bbox\_iou函数，然后将函数整个替换成下面的代码：

```java
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
    
 
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIoU=False, WIoU=False, Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7):
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
    iou = torch.pow(inter/(union + eps), alpha) # alpha iou
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIou
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
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha) # SIou
            elif WIoU:
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((rho2 / c2)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2)) # WIoU v1
            if Focal:
                return iou - rho2 / c2, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:
                return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU
```

这里要注意几个问题：

（1）WIoU有三个版本，分别对应控制如下：

 *  None: origin v1
 *  True: monotonic FM v2
 *  False: non-monotonic FM v3

（2） WIoU是不支持Focal的。

#### 第②步 配置loss.py文件 

这里和上面有点不同，我们需要找到utils/loss.py文件夹下的\_\_call\_\_函数，替换iou和lbox

（loss\_iou）函数，调用bbox iou损失函数时，将WIoU设置为True即可：

```java
# WIoU
       iou = bbox_iou(pbox, tbox[i], WIoU=True, Focal=False, scale=True)
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/e2aa68aa3370a3ef8840a17e33c6bf40.png)

![](https://i-blog.csdnimg.cn/blog_migrate/9e5778fbfd215b056cdf38c033cd37a0.png)

> 如果出现：TypeError: unsupported operand type(s) for -: 'float' and 'tuple'， 这个报错（float和元组不能相减）
> 
> 可参考评论区@[Mute2022][] 同学的改正方法：
> 
> lbox += (1.0 - iou).mean()换成：  
> if isinstance(iou, tuple):  
> if len(iou) == 2:  
> lbox += (iou\[1\].detach().squeeze() \* (1 - iou\[0\].squeeze())).mean()  
> iou = iou\[0\].squeeze()  
> else:  
> lbox += (iou\[0\] \* iou\[1\]).mean()  
> iou = iou\[2\].squeeze()  
> else:  
> lbox += (1.0 - iou.squeeze()).mean() \# iou loss  
> iou = iou.squeeze()

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

![](https://i-blog.csdnimg.cn/blog_migrate/34154b2c38716b6eeec89e181c99cbb4.gif)


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
[IoU]: #%E2%98%80%EF%B8%8F%E4%B8%80%E3%80%81%E4%B8%8D%E5%90%8CIoU%E7%9A%84%E4%BB%8B%E7%BB%8D
[IoU_GIoU_DIoU_CIoU]: #%E2%98%80%EF%B8%8F%E4%BA%8C%E3%80%81IoU%E3%80%81GIoU%E3%80%81DIoU%E3%80%81CIoU
[EIoU]: #%E2%98%80%EF%B8%8F%E4%B8%89%E3%80%81EIoU
[3.1]: #3.1%20%E7%AE%80%E4%BB%8B
[3.2]: #3.2%20%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[_metric.py]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%C2%A0%E9%85%8D%E7%BD%AEmetric.py%E6%96%87%E4%BB%B6
[_loss.py]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%C2%A0%E9%85%8D%E7%BD%AEloss.py%E6%96%87%E4%BB%B6
[Alpha IoU]: #%E2%98%80%EF%B8%8F%E5%9B%9B%E3%80%81Alpha%20IoU
[4.1]: #4.1%20%E7%AE%80%E4%BB%8B
[4.2]: #4.2%20%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[SIoU]: #%E2%98%80%EF%B8%8F%E4%BA%94%E3%80%81SIoU
[5.1]: #5.1%20%E7%AE%80%E4%BB%8B
[5.2]: #5.2%20%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[WIoU]: #%E2%98%80%EF%B8%8F%E5%85%AD%E3%80%81WIoU
[6.1]: #6.1%20%E7%AE%80%E4%BB%8B
[6.2]: #6.2%20%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[IoU_GIoU_DIoU_CIoU_EIoU_alpha IoU_SIoU_WIoU_Pytorch]: https://blog.csdn.net/weixin_43334693/article/details/131304963?spm=1001.2014.3001.5501
[Mute2022]: https://blog.csdn.net/Mute2022
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