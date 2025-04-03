#### ![](https://i-blog.csdnimg.cn/blog_migrate/5ba45e7bdf3c689b938aac99d86f17b2.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/4a7d26b7e1548714077bb435fae56cb0.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

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

[YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）][YOLOv5_17_IoU_MPDIoU_ELSEVIER 2023_WIoU_EIoU]

[YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）][YOLOv5_18_Neck_AFPN_PAFPN]

[YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）][YOLOv5_19_Swin TransformerV1_ViT]

[YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）][YOLOv5_20_BiFormer_CVPR2023]

[YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）][YOLOv5_21_RepViT_ ICCV 2023_ViT]

[YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）][YOLOv5_22_MobileViTv1_ ViT][YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）][YOLOv5_23_MobileViTv2_ Transformer]

![](https://i-blog.csdnimg.cn/blog_migrate/f31479040dad120f2e7e1c099fd38a2b.gif)

目录

[🚀一、CBAM注意力机制原理][CBAM]

[1.1 CBAM方法介绍][1.1 CBAM]

[1.2 通道注意力机制模块（CAM）][1.2 _CAM]

[1.3 空间注意力机制模块（SAM）][1.3 _SAM]

[1.4 具体过程][1.4]

[🚀二、添加CBAM注意力机制方法（单独加）][CBAM 1]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加CBAM模块][common.py_CBAM]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀三、添加C3\_CBAM注意力机制方法（在C3模块中添加）][C3_CBAM_C3]

[第①步：在common.py中添加CBAMBottleneck和C3\_CBAM模块][common.py_CBAMBottleneck_C3_CBAM]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/abeb07169a1cbdca6da5bc7bcfd42c44.gif)

## 🚀一、CBAM注意力机制原理 

> 论文题目：《CBAM: Convolutional Block Attention Module》
> 
> 论文地址：[https://arxiv.org/pdf/1807.06521.pdf][https_arxiv.org_pdf_1807.06521.pdf]
> 
> 代码实现：[CBAM.PyTorch][]

![](https://i-blog.csdnimg.cn/blog_migrate/675d77544168d0f6310e2c530178c8be.png)

### 1.1 CBAM方法介绍 

CBAM注意力机制是由通道注意力机制（channel）和空间注意力机制（spatial）组成。

![](https://i-blog.csdnimg.cn/blog_migrate/27a9a5d99624a04be0882a868cbd46f8.png)

在上一篇的SE中，我们学习了通道注意力机制（channel），而本篇的CBAM从通道channel 和 空间spatial 两个作用域出发，实现从通道到空间的顺序注意力结构。空间注意力可使神经网络更加关注图像中对分类起决定作用的像素区域而忽略无关紧要的区域，通道注意力则用于处理特征图通道的分配关系，同时对两个维度进行注意力分配增强了注意力机制对模型性能的提升效果。

### 1.2 通道注意力机制模块（CAM） 

![](https://i-blog.csdnimg.cn/blog_migrate/5813f5064eed4dddc0f75b21f58c56e8.png)

具体流程如下：  
首先，将输入的特征图F（H×W×C）分别经过基于width和height的最大池化和平均池化，对特征映射基于两个维度压缩，得到两个1×1×C的特征图

接着，再将最大池化和平均池化的结果利用共享的全连接层（Shared MLP）进行处理，先通过一个全连接层下降通道数，再通过另一个全连接恢复通道数。

然后，将共享的全连接层所得到的结果进行相加再使用Sigmoid激活函数，进而生成最终的channel attention feature，即获得输入特征层每一个通道的权重（0~1之间）。

最后，将权重通过乘法逐通道加权到输入特征层上，生成Spatial attention模块需要的输入特征。

代码实现如下： 

```java
#（1）通道注意力机制
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()
        
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel//ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel//ratio, out_features=in_channel, bias=False)
        
        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        
        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)
 
        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b,c])
        avg_pool = avg_pool.view([b,c])
 
        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)
 
        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)
        
        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)
        
        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x
        
        return outputs
```

### 1.3 空间注意力机制模块（SAM） 

![](https://i-blog.csdnimg.cn/blog_migrate/a9fe32e0ab697d801e4c5d199c6c04eb.png)

具体流程如下：  
将上面CAM模块输出的特征图F’作为本模块的输入特征图。

首先，对输入特征图在通道维度下做最大池化和平均池化，将池化后的两张特征图在通道维度堆叠（concat）。

然后，经过一个7×7卷积（7×7比3×3效果要好）操作，降维为1个channel，即卷积核融合通道信息，特征图的shape从 \[b,2,h,w\] 变成 \[b,1,h,w\]。

最后，将卷积后的结果经过 sigmoid 函数对特征图的空间权重归一化，再将输入特征图和权重相乘。

代码实现如下： 

```java
#（2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()
        
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播
    def forward(self, inputs):
        
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x
        
        return outputs
```

### 1.4 具体过程 

CBAM的操作过程分为CAM和SAM，给出一个中间特征图![F\in R^{C\times H\times W}](https://latex.csdn.net/eq?F%5Cin%20R%5E%7BC%5Ctimes%20H%5Ctimes%20W%7D)作为输入。

首先，对输入通道进行全局最大池化和全局平均池化，将池化后的两个一维向量输入全连通层，相加生成一维信道注意![M_{c}\in R^{C\times H\times W}](https://latex.csdn.net/eq?M_%7Bc%7D%5Cin%20R%5E%7BC%5Ctimes%20H%5Ctimes%20W%7D)，然后将信道注意与输入元素相乘，得到信道注意调整特征图F’。

其次，F是由全球最大和平均池空间，和池生成的两个二维向量池拼接和卷积最终生成二维空间关注![M_{s}\in R^{C\times H\times W}](https://latex.csdn.net/eq?M_%7Bs%7D%5Cin%20R%5E%7BC%5Ctimes%20H%5Ctimes%20W%7D)，并将空间注意力与F的元素，如图2所示。注意过程可以用以下公式(1)和公式(2)来描述。

![](https://i-blog.csdnimg.cn/blog_migrate/1450f33fad17b5167b01f81a9bf85197.png)

其中，⊗表示对应的元素乘法，在乘法操作前，需要分别根据空间维度和信道维度来广播信道注意和空间注意。通道注意方程为：

![](https://i-blog.csdnimg.cn/blog_migrate/6fe5533fd7a4a0e10aacdc0b3683672a.png) 空间注意序列是：

![](https://i-blog.csdnimg.cn/blog_migrate/5f0dd40ff74de691db9fe29ee23e981d.png)

CBAM模块结构如下：

![](https://i-blog.csdnimg.cn/blog_migrate/268c5686b879a94e05c005ab3a2bc41b.png)

## 🚀二、添加CBAM注意力机制方法（单独加） 

### 2.1 添加顺序 

（1）models/common.py -->  加入新增的网络结构

（2） models/yolo.py -->  设定网络结构的传参细节，将CBAM类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml -->  新建一个文件夹，如yolov5s\_CBAM.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加CBAM模块 

将下面的CBAM代码复制粘贴到common.py文件的末尾

```java
# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/05b8402137670360357de4b386681edc.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

然后把CBAM添加到这个注册表里面

![](https://i-blog.csdnimg.cn/blog_migrate/1a42cbb90f324629f0c1dc4169c75780.png)

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml文件，粘贴并重命名为yolov5s\_CBAM.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/d8dfc7c163e08da1fa05f806706f182c.png)

接着修改 yolov5s\_CBAM.yaml ，将CBAM模块加到我们想添加的位置。

注意力机制可以添加在backbone，Neck，Head等部分， 常见的有两种：一是在主干的 SPPF 前添加一层；二是将Backbone中的C3全部替换。

在这里我是用第一种：将 \[-1，1，CBAM，\[1024\]\]添加到 SPPF 的上一层，下一节使用第二种。即下图中所示位置：

![](https://i-blog.csdnimg.cn/blog_migrate/f751023d541a93941207d05845ba9464.png)

同样的下面的head也得修改，p4，p5以及最后detect的总层数都得+1

![](https://i-blog.csdnimg.cn/blog_migrate/78edc7ac33a500e4734c8d4c80ea8610.png)

这里我们要把后面两个Concat的from系数分别由\[ − 1 , 14 \] , \[ − 1 , 10 \]改为\[ − 1 , 15 \], \[ − 1 , 11 \]。然后将Detect原始的from系数\[ 17 , 20 , 23 \]要改为\[ 18 , 21 , 24 \] 。

![](https://i-blog.csdnimg.cn/blog_migrate/3fbb39efeb5b506a42408c36b2bffa33.png)

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_CBAM.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/7f2d039f25b9ca2b32fb13b2446692a9.png)

![](https://i-blog.csdnimg.cn/blog_migrate/f3085b54899fbce1c877cb99f3d632e0.png)

然后运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/cba02ebaa5f72e30fb01911cab64a169.png)

找到CBAM这一层，就说明我们添加成功啦！

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py 文件的parse\_opt函数，然后将第二行‘--cfg’的default改为'models/yolov5s\_CBAM.yaml'，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/69f7f3e6b2b6b381775c14587248c306.png)

## 🚀三、添加C3\_CBAM注意力机制方法（在C3模块中添加） 

上面是单独加注意力层，接下来的方法是在C3模块中加入注意力层。

刚才也提到了，这个策略是将CBAM注意力机制添加到Bottleneck，替换Backbone中的所有C3模块。

（因为步骤和上面相同，所以接下来只放重要步骤噢~）

#### 第①步：在common.py中添加CBAMBottleneck和C3\_CBAM模块 

将下面的代码复制粘贴到common.py文件的末尾

```java
# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)

class CBAMBottleneck(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion, ratio, kernel_size
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16, kernel_size=7):
        super(CBAMBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # 加入CBAM模块
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # 考虑加入CBAM模块的位置：bottleneck模块刚开始时、bottleneck模块中shortcut之前，这里选择在shortcut之前
        x2 = self.cv2(self.cv1(x))  # x和x2的channel数相同
        # 在bottleneck模块中shortcut之前加入CBAM模块
        out = self.channel_attention(x2) * x2
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return x + out if self.add else out


class C3_CBAM(C3):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

在yolo.py的`parse_model`函数中，加入`CBAMBottleneck, C3_CBAM这`两个模块

![](https://i-blog.csdnimg.cn/blog_migrate/3ca1537acc57c12373e16b67d2466ded.png)

#### 第③步：创建自定义的yaml文件 

按照上面的步骤创建yolov5s\_C3\_CBAM.yaml文件，替换4个C3模块

![](https://i-blog.csdnimg.cn/blog_migrate/9516dd68cb81d344addedb667b61749a.png)

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3_CBAM, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3_CBAM, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 3, C3_CBAM, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3_CBAM, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

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

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_C3\_CBAM.yaml，然后运行

![](https://i-blog.csdnimg.cn/blog_migrate/e8235c49e950016d0f200a85cbbaae1e.png)

这样就OK啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

接下来的训练就和上面一样，不再叙述啦~

完结~撒花✿✿ヽ(°▽°)ノ✿

PS：加入不同的位置效果不同，这两个我各训练100轮看了下效果，对于我的数据集来说，第2种比第1种mAP增加0.009。

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

> 本文参考（感谢大佬们）：
> 
> b站：[【YOLOv5 v6.1添加SE,CA,CBAM,ECA注意力机制教学，即插即用】][YOLOv5 v6.1_SE_CA_CBAM_ECA]
> 
> CSDN： [【深度学习】(1) CNN中的注意力机制（SE、ECA、CBAM）\_立Sir的博客-CSDN博客][1_ CNN_SE_ECA_CBAM_Sir_-CSDN]
> 
> [(强推) 手把手带你YOLOv5 (v6.1)添加注意力机制(二)（在C3模块中加入注意力机制）\_迪菲赫尔曼的博客-CSDN博客][_YOLOv5 _v6.1_C3_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/2a94f52802c04e1a86599e33518ad91d.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
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
[YOLOv5_20_BiFormer_CVPR2023]: https://blog.csdn.net/weixin_43334693/article/details/132203200?spm=1001.2014.3001.5502
[YOLOv5_21_RepViT_ ICCV 2023_ViT]: https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501
[YOLOv5_22_MobileViTv1_ ViT]: https://blog.csdn.net/weixin_43334693/article/details/132367429?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22132367429%22%2C%22source%22%3A%22weixin_43334693%22%7D
[YOLOv5_23_MobileViTv2_ Transformer]: https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5501
[CBAM]: #%F0%9F%9A%80%E4%B8%80%E3%80%81CBAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E5%8E%9F%E7%90%86
[1.1 CBAM]: #1.1%20CBAM%E6%96%B9%E6%B3%95%E4%BB%8B%E7%BB%8D
[1.2 _CAM]: #1.2%C2%A0%E9%80%9A%E9%81%93%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%A8%A1%E5%9D%97%EF%BC%88CAM%EF%BC%89
[1.3 _SAM]: #1.3%C2%A0%E7%A9%BA%E9%97%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%A8%A1%E5%9D%97%EF%BC%88SAM%EF%BC%89
[1.4]: #1.4%20%E5%85%B7%E4%BD%93%E8%BF%87%E7%A8%8B
[CBAM 1]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%B7%BB%E5%8A%A0CBAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%8D%95%E7%8B%AC%E5%8A%A0%EF%BC%89
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_CBAM]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[C3_CBAM_C3]: #%C2%A0%F0%9F%9A%80%E4%B8%89%E3%80%81%E6%B7%BB%E5%8A%A0C3_CBAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95%EF%BC%88%E5%9C%A8C3%E6%A8%A1%E5%9D%97%E4%B8%AD%E6%B7%BB%E5%8A%A0%EF%BC%89
[common.py_CBAMBottleneck_C3_CBAM]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0CBAMBottleneck%E5%92%8CC3_CBAM%E6%A8%A1%E5%9D%97
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[https_arxiv.org_pdf_1807.06521.pdf]: https://arxiv.org/pdf/1807.06521.pdf
[CBAM.PyTorch]: https://github.com/luuuyi/CBAM.PyTorch
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
[YOLOv5 v6.1_SE_CA_CBAM_ECA]: /https://www.bilibili.com/video/BV1kS4y1c7Bm?vd_source=725f2b2a52500df1eaed63206ebe0ab2
[1_ CNN_SE_ECA_CBAM_Sir_-CSDN]: https://blog.csdn.net/dgvv4/article/details/125112972?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168350459816800225558052%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168350459816800225558052&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-5-125112972-null-null.142%5Ev86%5Ekoosearch_v1,239%5Ev2%5Einsert_chatgpt&utm_term=%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187
[_YOLOv5 _v6.1_C3_-CSDN]: https://blog.csdn.net/weixin_43694096/article/details/124695537?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22124695537%22%2C%22source%22%3A%22weixin_43694096%22%7D