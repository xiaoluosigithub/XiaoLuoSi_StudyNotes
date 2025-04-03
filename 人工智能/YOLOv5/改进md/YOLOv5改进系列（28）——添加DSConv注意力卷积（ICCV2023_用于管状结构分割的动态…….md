![](https://i-blog.csdnimg.cn/blog_migrate/ab833e167e45b1101669106e3d5b8b5d.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/38dd9fc70e8e4ebe778145f36dde11d1.png)

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

[YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等][YOLOv5_13_SiLU_ReLU_ELU_Hardswish_Mish_Softplus_AconC][YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS][YOLOv5_14_NMS_ DIoU-NMS_CIoU-NMS_EIoU-NMS_GIoU-NMS _SIoU-NMS_Soft-NMS]

[YOLOv5改进系列（15）——增加小目标检测层][YOLOv5_15]

[YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）][YOLOv5_16_EMA_ICASSP2023]

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT]

[YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）][YOLOv5_23_MobileViTv2_ Transformer]

[YOLOv5改进系列（24）——替换主干网络之MobileViTv3（移动端轻量化网络的进一步升级）][YOLOv5_24_MobileViTv3]

[YOLOv5改进系列（25）——添加LSKNet注意力机制（大选择性卷积核的领域首次探索）][YOLOv5_25_LSKNet]

[YOLOv5改进系列（26）——添加RFAConv注意力卷积（感受野注意力卷积运算）][YOLOv5_26_RFAConv]

[YOLOv5改进系列（27）——添加SCConv注意力卷积（CVPR 2023|即插即用的高效卷积模块）][YOLOv5_27_SCConv_CVPR 2023]

![](https://i-blog.csdnimg.cn/blog_migrate/87e1f45ae909f6c51cbb0ddb973a19b6.gif)

目录

[🚀 一、DSConv介绍 ][_DSConv_]

[1.1 DSConv简介][1.1 DSConv]

[ 1.2 动态蛇形卷积][1.2]

[1.3 多视角特征融合策略][1.3]

[ 1.4 连续性拓扑约束损失][1.4]

[🚀二、具体添加方法][Link 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加DCConv模块 ][common.py_DCConv_]

[第②步：修改yolo.py文件 ][yolo.py_]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/70c24b84eba53cad07e275ce219f0ac2.gif)

## 🚀 一、DSConv介绍 

> 学习资料：
> 
>  *  论文题目：《Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation》
>  *  论文地址：[https://arxiv.org/abs/2307.08388][https_arxiv.org_abs_2307.08388]
>  *  源码地址：[https://github.com/YaoleiQi/DSCNet][https_github.com_YaoleiQi_DSCNet]

### 1.1 DSConv简介 

背景

管状结构（例如血管、道路）是临床、自然界等各领域场景中十分重要的一种结构，其精确分割可以保证下游任务的准确性与效率。但管状结构的精确提取仍然面临着众多挑战：

 *  细长且脆弱的局部结构。如下图所示，细长的结构仅占整个图像的一小部分，像素的组成有限。此外，这些结构容易受到复杂背景的干扰，因此模型很难精确分辨目标的细微变化，从而导致分割出现破碎与断裂。
 *  复杂且多变的全局形态。如下图所示，我们可以看出细长管状结构复杂多变的形态，即使在同一张图像中也是如此。位于不同区域的目标的形态变化取决于分支的数量、分叉的位置，路径长度以及其在图像中的位置。因此当数据表现出未曾见过的形态特征时，模型倾向于过拟合到已见过的特征，无法识别未见过的特征形态，从而导致泛化性较弱。

![](https://i-blog.csdnimg.cn/blog_migrate/e3567b4edaab39eb9f6cd7afbdd68eb0.png)

本文主要工作 

本文关注到管状结构细长连续的特点，并利用这一信息在神经网络以下三个阶段同时增强感知：特征提取、特征融合和损失约束。分别设计了动态蛇形卷积（Dynamic Snake Convolution），多视角特征融合策略与连续性拓扑约束损失。我们同时给出了基于 2D 和 3D 的方法设计，通过实验证明了本文所提出的 DSCNet 在管状结构分割任务上提供了更好的精度和连续性。

### 1.2 动态蛇形卷积 

![](https://i-blog.csdnimg.cn/blog_migrate/f454d875df0ccb933c480fd7708d3910.png)

目的：

 *  希望卷积核一方面能够自由地贴合结构学习特征
 *  另一方面能够在约束条件下不偏离目标结构太远

可变形卷积：

 *  操控单个卷积核形变的所有偏置(offset)，是在网络中一次性全部学到的
 *  对于这一个偏置只有一个范围的约束，即感受野范围(extend)
 *  控制所有的卷积发生形变，是依赖于整个网络最终的损失约束回传，这个变化过程是相当自由的。

### 1.3 多视角特征融合策略 

![](https://i-blog.csdnimg.cn/blog_migrate/700cab0a073cdb4fbfd72f5e8c628043.png)

目的：

 *  管状结构的走向与视角不是单一的，因此在设计中融合多视角特征也是必然的选择。

挑战：

 *  融合更多的特征会导致更大的网络负载以及出现冗余。

方法：

 *  在特征融合的训练过程中加入了分组与dropout的策略，一定程度上缓解了网络内内存的压力并避免模型陷入过拟合。

### 1.4 连续性拓扑约束损失 

![](https://i-blog.csdnimg.cn/blog_migrate/b3d0df27308f93b26e182819d6dabd4b.png)

目的：

 *  构建数据的拓扑结构，并提取复杂管状结构中的高维关系，也就是持续同源性（Persistence Homology, PH）。

启发：

 *  假设 PO 的上端存在着一个异常的离散点（横坐标表示出现的时间，纵坐标表示消失的时间），这表明存在一个构件直到最后才与其他构件获得连接从而消失。

方法： 

 *  本文中采用的是豪斯多夫距离（Hausdorff Distance, HD），HD 也是用于衡量点集相似度的一个重要算法，对离散点也非常敏感。

```java
# -*- coding: utf-8 -*-

import torch

from torch import nn
from torch.nn.functional import max_pool3d


class crossentry(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))


class cross_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))


'''
Another Loss Function proposed by us in IEEE transactions on Image Precessing:
Paper: https://ieeexplore.ieee.org/abstract/document/9611074
Code: https://github.com/YaoleiQi/Examinee-Examiner-Network
'''


class Dropoutput_Layer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = (
            -((torch.sum(w * y_true * torch.log(y_pred + smooth)) /
               torch.sum(w * y_true + smooth)) +
              (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) /
               torch.sum(w * (1 - y_true) + smooth))) / 2)
        return loss_ce
```

## 🚀二、具体添加方法 

### 2.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将DSConv类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_DSConv.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py -->  修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加DCConv模块  

将下面的DSConv代码复制粘贴到common.py文件的末尾。

```java
# by:迪菲赫尔曼
import warnings
import torch
from torch import nn

warnings.filterwarnings("ignore")

"""
This code is mainly the deformation process of our DSConv
"""

class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset):
        """
        动态蛇形卷积
        :param in_ch: 输入通道
        :param out_ch: 输出通道
        :param kernel_size: 卷积核的大小
        :param extend_scope: 扩展范围（默认为此方法的1）
        :param morph: 卷积核的形态主要分为两种类型，沿x轴（0）和沿y轴（1）（详细信息请参阅论文）
        :param if_offset: 是否需要变形，如果为False，则是标准卷积核
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        # offset_conv: 学习可变形偏移的卷积层
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        # dsc_conv_x 和 dsc_conv_y：两种动态蛇形卷积层，分别沿x轴和y轴。
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )
        
        # gn：组归一化层
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        # extend_scope：扩展范围
        self.extend_scope = extend_scope
        # morph：卷积核形态的类型
        self.morph = morph
        # if_offset：指示是否需要变形的布尔值
        self.if_offset = if_offset

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature.type(f.dtype))
            x = self.gn(x)
            x = self.relu(x)
            return x


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):

    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    """
    input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
    output_x: [B,1,W,K*H]   coordinate map
    output_y: [B,1,K*W,H]   coordinate map
    """

    def _coordinate_map_3D(self, offset, if_offset):
        """
        1.输入为偏移 (offset) 和是否需要偏移 (if_offset)。
        2.根据输入特征图的形状、卷积核大小、扩展范围以及形态类型，生成二维坐标映射。
        3.如果形态类型为0，表示沿x轴，生成y坐标映射；如果形态类型为1，表示沿y轴，生成x坐标映射。
        4.根据偏移和扩展范围调整坐标映射。
        5.返回生成的坐标映射。
        """
        device = offset.device
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(device)
            x_new = x_new.to(device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    """
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    """

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        """
         1.输入为输入特征图 (input_feature)、y坐标映射 (y) 和x坐标映射 (x)。
         2.进行三维双线性插值，获取变形后的特征。
         3.返回插值得到的变形特征。
        """
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        """
        1.输入为原始特征图 (input)、偏移 (offset) 和是否需要偏移 (if_offset)。
        2.调用 _coordinate_map_3D 方法获取坐标映射。
        3.调用 _bilinear_interpolate_3D 方法进行双线性插值，得到变形后的特征。
        4.返回变形后的特征。
        """
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

#---------------------------------YOLOv5 专用部分↓---------------------------------
class DSConv_Bottleneck(nn.Module):
    # DSConv bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.snc = DSConv(c2, c2, 3, 1, 1, True)

    def forward(self, x):
        return x + self.snc(self.cv2(self.cv1(x))) if self.add else self.snc(self.cv2(self.cv1(x)))


class DSConv_C3(nn.Module):
    # DSConv Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(DSConv_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

#---------------------------------YOLOv5 专用部分↑---------------------------------
```

#### 第②步：修改yolo.py文件 

再来修改yolo.py，在parse\_model函数中找到elif m is nn.BatchNorm2d:语句，在其后面加上下面代码：

```java
elif m in (DSConv, DSConv_C3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is DSConv_C3:
                args.insert(2, n)  # number of repeats
                n = 1
```

 如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/c93b63a00ffc98d897e29f5d16cb3f2f.png)

#### 第③步：创建自定义的yaml文件 

第1种，替换conv结构

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.5  # layer channel multiple
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
 
# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512]],  # 13
 
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, DSConv, [256, 3,1,1,True]],  # 17 (P3/8-small)
 
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, DSConv, [512, 3,1,1,True]],  # 20 (P4/16-medium)
 
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, DSConv, [1024, 3,1,1,True]],  # 23 (P5/32-large)
 
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

这里要注意一个问题，替换时DSConv参数是需要做对应修改：

![](https://i-blog.csdnimg.cn/blog_migrate/4f57b7fb0a0323428a9cd3675ebea643.png)

如下图栗子所示：

![](https://i-blog.csdnimg.cn/blog_migrate/8aeb522907fff4e8ffb2117828fe3ed5.png)

如果直接改模块名会出现缺参报错：

```java
TypeError: __init__() missing 2 required positional arguments: 'morph' and 'if_offset'
```

![](https://i-blog.csdnimg.cn/blog_migrate/64366f3d8ee26b4ef75e0cedd64fd7b8.png)

第2种，替换C3模块

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.5  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, DSConv_C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, DSConv_C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, DSConv_C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, DSConv_C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]
 
# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512]],  # 13
 
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
 
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
 
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
 
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

替换C3模块直接改模块名字就行。

> 代码参考：
> 
> [改进YOLO系列 | YOLOv5/v7 引入 Dynamic Snake Convolution | 动态蛇形卷积\_yolov7加入dynamic snake convolution-CSDN博客][YOLO_ _ YOLOv5_v7 _ Dynamic Snake Convolution _ _yolov7_dynamic snake convolution-CSDN]

#### 第④步：验证是否加入成功 

运行yolo.py

第1种

![](https://i-blog.csdnimg.cn/blog_migrate/77a59ddd31bb07bab8174a56468cdfa6.png)

第2种

![](https://i-blog.csdnimg.cn/blog_migrate/b9e5127627db98c4116f4fb5cf6042b2.png) 这样就OK啦！

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

![](https://i-blog.csdnimg.cn/blog_migrate/2c2007740e861597f69f381c445a3199.gif)


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
[YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]: https://blog.csdn.net/weixin_43334693/article/details/131999141?spm=1001.2014.3001.5501
[YOLOv5_18_Neck_AFPN_PAFPN]: https://blog.csdn.net/weixin_43334693/article/details/132070079?spm=1001.2014.3001.5501
[YOLOv5_19_Swin TransformerV1_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132161488?spm=1001.2014.3001.5501
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132367429%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502
[YOLOv5_24_MobileViTv3]: https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502
[YOLOv5_25_LSKNet]: https://blog.csdn.net/weixin_43334693/article/details/135510571?spm=1001.2014.3001.5501
[YOLOv5_26_RFAConv]: https://blog.csdn.net/weixin_43334693/article/details/135562865?spm=1001.2014.3001.5501
[YOLOv5_27_SCConv_CVPR 2023]: https://blog.csdn.net/weixin_43334693/article/details/135610505?spm=1001.2014.3001.5502
[_DSConv_]: #%F0%9F%9A%80%C2%A0%E4%B8%80%E3%80%81DSConv%E4%BB%8B%E7%BB%8D%C2%A0
[1.1 DSConv]: #1.1%C2%A0DSConv%E7%AE%80%E4%BB%8B
[1.2]: #%C2%A01.2%C2%A0%E5%8A%A8%E6%80%81%E8%9B%87%E5%BD%A2%E5%8D%B7%E7%A7%AF
[1.3]: #1.3%C2%A0%C2%A0%E5%A4%9A%E8%A7%86%E8%A7%92%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88%E7%AD%96%E7%95%A5
[1.4]: #%C2%A01.4%C2%A0%E8%BF%9E%E7%BB%AD%E6%80%A7%E6%8B%93%E6%89%91%E7%BA%A6%E6%9D%9F%E6%8D%9F%E5%A4%B1
[Link 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_DCConv_]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0LSK%E6%A8%A1%E5%9D%97%C2%A0
[yolo.py_]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9yolo.py%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[yaml_]: #%C2%A0%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0%C2%A0
[Link 2]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_abs_2307.08388]: https://arxiv.org/abs/2307.08388
[https_github.com_YaoleiQi_DSCNet]: https://github.com/YaoleiQi/DSCNet
[YOLO_ _ YOLOv5_v7 _ Dynamic Snake Convolution _ _yolov7_dynamic snake convolution-CSDN]: https://yolov5.blog.csdn.net/article/details/133973402
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