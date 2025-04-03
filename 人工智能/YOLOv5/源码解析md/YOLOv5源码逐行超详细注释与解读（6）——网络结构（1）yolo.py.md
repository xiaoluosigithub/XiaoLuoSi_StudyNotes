#### ![](https://i-blog.csdnimg.cn/blog_migrate/4801687a95a03dac27aa577ea48baf7d.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/e4fd097b3684a09db939cd2d3edec21a.jpeg)

## 前言 

在上一篇中，我们简单介绍了YOLOv5的配置文件之一yolov5s.yaml，这个文件中涉及很多参数，它们的调用会在这篇yolo.py 和下一篇common.py 中具体实现。

本篇我们会介绍 yolo.py，这是YOLO的特定模块，和网络构建有关。在 YOLOv5源码中，模型的建立是依靠 yolo.py 中的函数和对象完成的，这个文件主要由三个部分：parse\_model函数、Detect类和Model类组成。

yolo.py文件位置在./models/yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/c199741acd4b417021766d4d2081ac3e.png)

文章代码逐行手打注释，每个模块都有对应讲解，一文帮你梳理整个代码逻辑！

友情提示：全文4万字，可以先点![](https://i-blog.csdnimg.cn/blog_migrate/ea5f7225888a49f6a6827b9ae71e856f.gif)再慢慢看哦~

源码下载地址：[mirrors / ultralytics / yolov5 · GitCode][mirrors _ ultralytics _ yolov5 _ GitCode]

![](https://i-blog.csdnimg.cn/blog_migrate/a94e59d738b3fcf5113fc5a825b4fa21.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

## 目录 

[前言][Link 1]

[🚀一、 导包和基本配置][Link 2]

[1.1 导入安装好的python库 ][1.1 _python_]

[1.2 获取当前文件的绝对路径][1.2]

[1.3 加载自定义模块][1.3]

[🚀二、parse\_model函数][parse_model]

[2.1 获取对应参数][2.1]

[2.2 搭建网络前准备][2.2]

[2.3 更新当前层的参数，计算c2][2.3 _c2]

[2.4 使用当前层的参数搭建当前层][2.4]

[2.5 打印和保存layers ][2.5 _layers]

[🚀 三、Detect模块][_Detect]

[3.1 获取预测得到的参数][3.1]

[3.2 向前传播][3.2]

[3.3 相对坐标转换到grid绝对坐标系][3.3 _grid]

[🚀四、Model类][Model]

[4.1 \_\_init\_\_函数][4.1 _init]

[4.2 数据增强相关函数][4.2]

[4.2.1 forward():管理前向传播函数][4.2.1 forward]

[4.2.2 \_forward\_augment():推理的forward][4.2.2 _forward_augment_forward]

[4.2.3 \_forward\_once():训练的forward][4.2.3 _forward_once_forward]

[4.2.4 \_descale\_pred():将推理结果恢复到原图尺寸][4.2.4 _descale_pred]

[4.2.5 \_clip\_augmented（）:TTA的时候对原图片进行裁剪][4.2.5 _clip_augmented_TTA]

[4.2.6 \_profile\_one\_layer（）:打印日志信息][4.2.6 _profile_one_layer]

[4.2.7 \_initialize\_biases（）:初始化偏置biases信息][4.2.7 _initialize_biases_biases]

[4.2.8 \_print\_biases（）:打印偏置biases信息][4.2.8 _print_biases_biases]

[4.2.9 fuse（）:将Conv2d+BN进行融合][4.2.9 fuse_Conv2d_BN]

[4.2.10 autoshape（）:扩展模型功能][4.2.10 autoshape]

[4.2.11 info():打印模型结构信息][4.2.11 info]

[4.2.12 \_apply():将模块转移到 CPU/ GPU上][4.2.12 _apply_ CPU_ GPU]

[🚀五、yolo.py全部注释][yolo.py]

## ![](https://i-blog.csdnimg.cn/blog_migrate/b120286603644573938f19dfa5700fe1.gif) 

## 🚀一、 导包和基本配置 

### 1.1 导入安装好的python库 

```java
'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from copy import deepcopy  # 数据拷贝模块 深拷贝
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
```

首先，导入一下常用的python库：

 *  argparse： 它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
 *  sys： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
 *  copy： Python 中赋值语句不复制对象，而是在目标和对象之间创建绑定关系。copy模块提供了通用的浅层复制和深层复制操作
 *  pathlib： 这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读

### 1.2 获取当前文件的绝对路径 

```java
'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve() # __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/modles/yolo.py
ROOT = FILE.parents[1]  # YOLOv5 root directory 保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
# ROOT = ROOT.relative_to(Path.cwd())  # relative  ROOT设置为相对路径
```

这段代码会获取当前文件的绝对路径，并使用Path库将其转换为Path对象。

这一部分的主要作用有两个：

 *  将当前项目添加到系统路径上，以使得项目中的模块可以调用。
 *  将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

### 1.3 加载自定义模块 

```java
'''===================3..加载自定义模块============================'''
from models.common import * # yolov5的网络结构(yolov5)
from models.experimental import *   # 导入在线下载模块
from utils.autoanchor import check_anchor_order   # 导入检查anchors合法性的函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args   # 定义了一些常用的工具函数
from utils.plots import feature_visualization  # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)   # 定义了一些与PyTorch有关的工具函数

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
```

这些都是用户自定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换。具体来说，代码从如下几个文件中导入了部分函数和类：

 *  models.common： 这个是yolov5的网络结构
 *  models.experimental： 实验性质的代码，包括MixConv2d、跨层权重Sum等
 *  utils.autoanchor： 定义了自动生成锚框的方法
 *  utils.general： 定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
 *  utils.plots：定义了Annotator类，可以在图像上绘制矩形框和标注信息
 *  utils.torch\_utils： 定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等

通过导入这些模块，可以更方便地进行目标检测的相关任务，并且减少了代码的复杂度和冗余。

## 🚀二、parse\_model函数 

parse\_model函数用在DetectionModel模块中，主要作用是解析模型yaml的模块，通过读取yaml文件中的配置，并且到common.py中找到相对于的模块，然后组成一个完整的模型解析模型文件(字典形式)，并搭建网络结构。简单来说，就是把yaml文件中的网络结构实例化成对应的模型。后续如果需要动模型框架的话，需要对这个函数做相应的改动。

### 2.1 获取对应参数 

```java
def parse_model(d, ch):  # model_dict, input_channels(3)

    '''===================1. 获取对应参数============================'''

    # 使用 logging 模块输出列标签
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 获取anchors，nc，depth_multiple，width_multiple
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: na * 属性数 (5 + 分类数)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
```

这段代码主要是获取配置dict里面的参数，并打印最开始展示的网络结构表的表头。

我们先解释几个参数，d和ch，na和no：

 *   d: yaml 配置文件（字典形式），yolov5s.yaml中的6个元素 + ch
 *  ch: 记录模型每一层的输出channel，初始ch=\[3\]，后面会删除
 *  na： 判断anchor的数量
 *  no： 根据anchor数量推断的输出维度

这里有一行代码我们上篇[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]就见过了：

![](https://i-blog.csdnimg.cn/blog_migrate/955ff1b89fcaa4c1414990bc6a62daa7.png)

这里就是读取了 yaml 文件的相关参数（参数含义忘了的话再看看上篇哦）

### 2.2 搭建网络前准备 

```java
'''===================2. 搭建网络前准备============================'''
    # 网络单元列表, 网络输出引用列表, 当前的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 读取 backbone, head 中的网络单元
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 利用 eval 函数, 读取 model 参数对应的类名 如‘Focus’,'Conv'等
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 利用 eval 函数将字符串转换为变量 如‘None’,‘nc’，‘anchors’等
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass
```

这段代码主要是遍历backbone和head的每一层，获取搭建网络前的一系列信息。

我们还是先解释参数，layers、save和c2：

 *  layers:  保存每一层的层结构
 *  save: 记录下所有层结构中from不是-1的层结构序号
 *  c2:  保存当前层的输出channel 

然后开始迭代循环backbone与head的配置。for i, (f, n, m, args) in enumerate(d\['backbone'\] + d\['head'\]):中有几个参数

 *  f： from，当前层输入来自哪些层
 *  n： number，当前层次数 初定
 *  m： module，当前层类别
 *  args： 当前层类参数 初定

接着还用到一个函数eval（），主要作用是将字符串当成有效的表达式来求值，并且返回执行的结果。在这里简单来说，就是实现list、dict、tuple与str之间的转化。

### 2.3 更新当前层的参数，计算c2 

```java
'''===================3. 更新当前层的参数，计算c2============================'''
        # depth gain: 控制深度，如yolov5s: n*0.33
        # n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 当该网络单元的参数含有: 输入通道数, 输出通道数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # no=75，只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # width gain: 控制宽度，如yolov5s: c2*0.5; c2: 当前层的最终输出channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)
```

这段代码主要是更新当前层的args，计算c2(当前层的输出channel)

首先网络将C3中的BottleNeck数量乘以模型缩放倍数n\*gd控制模块的深度缩放，举个栗子，对于yolo5s来讲，gd为0.33，那么就是n\*0.33，也就是把默认的深度缩放为原来的1/3。

然后将m实例化成同名模块，别看列举了那么多模块，目前只用到Conv，SPP，Focus，C3，nn.Upsample。对于以上的这几种类型的模块，ch是一个用来保存之前所有的模块输出的channle，ch\[-1\]代表着上一个模块的输出通道。args\[0\]是默认的输出通道。

这样以来，c1=ch\[f\]就代表输入通道c1为f指向的层的输出通道，c2=args\[0\]就代表输出通道c2为yaml的args中的第一个变量。注意，如果输出通道不等于255即Detect层的输出通道， 则将通道数乘上width\_multiple，并调整为8的倍数。通过函数make\_divisible来实现

make\_divisible（）代码如下：

```java
   # 使得X能够被divisor整除
     def make_divisible(x, divisor):
         return math.ceil(x / divisor) * divisor
```

### 2.4 使用当前层的参数搭建当前层 

```java
'''===================4.使用当前层的参数搭建当前层============================'''
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channels, out_channels, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost/C3x，则需要在args中加入Bottleneck的个数
            # [in_channels, out_channels, Bottleneck个数, Bool(shortcut有无标记)]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                # 在第二个位置插入bottleneck个数n
                args.insert(2, n)  # number of repeats
                # 恢复默认值1
                n = 1
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        # 判断是否是tensor连接模块
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        # 判断是否是detect模块
        elif m is Detect:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors 几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract: # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:  # 不怎么用
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]  # args不变
```

这段代码主要是使用当前层的参数搭建当前层。

经过以上处理，args里面保存的前两个参数就是module的输入通道数、输出通道数。只有BottleneckCSP和C3这两种module会根据深度参数n调整该模块的重复迭加次数。

然后进行的是其他几种类型的Module判断：

 *  如果是BN层，只需要返回上一层的输出channel，通道数保持不变。
 *  如果是Concat层，则将f中所有的输出累加得到这层的输出channel，f是所有需要拼接层的index，输出通道c2是所有层的和。
 *  如果是Detect层，则对应检测头部分，这块下一小节细讲。

Contract和Expand目前未在模型中使用。

### 2.5 打印和保存layers 

```java
'''===================5.打印和保存layers信息============================'''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print

        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
```

这段代码主要是打印当前层结构的一些基本信息并保存。

把构建的模块保存到layers里，把该层的输出通道数写入ch列表里。待全部循环结束后再构建成模型。

返回值：

 *  return nn.Sequential(\*layers):  网络的每一层的层结构
 *  return sorted(save):  把所有层结构中from不是-1的值记下 并排序 \[4, 6, 10, 14, 17, 20, 23\]

至此模型就全部构建完毕了。

下面详细介绍一下各个模块。

## 🚀 三、Detect模块 

Detect 模块是 YOLO 网络模型的最后一层 (对应 yaml 文件最后一行)，通过 yaml 文件进行声明，格式为：

```java
[*from], 1, Detect, [nc, anchors]
```

其中 nc 为分类数，anchors 为先验框，修改 yaml 文件的前几行即可。

在 parse\_model 函数中，会根据 from 参数，找到对应网络层的输出通道数。传参给 Detect 对象后，生成对应的 Conv2d，为后面的计算损失或者NMS后处理作准备。

### 3.1 获取预测得到的参数 

```java
class Detect(nn.Module):
    stride = None  # 特征图的缩放步长
    onnx_dynamic = False  # ONNX动态量化

    '''===================1.获取预测得到的参数============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
     
        super().__init__()

        # nc: 数据集类别数量
        self.nc = nc
        # no: 表示每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # nc+5=nc+(x,y,w,h,conf)
        # nl: 表示预测层数，yolov5是3层预测
        self.nl = len(anchors)
        # na: 表示anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2
        # grid: 表示初始化grid列表大小，下面会计算grid，grid就是每个格子的x，y坐标（整数，比如0-19），左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        self.grid = [torch.zeros(1)] * self.nl
        # anchor_grid: 表示初始化anchor_grid列表大小，空列表
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2) 注意后面就可以通过self.anchors来访问它了
        # 每一张进行三次预测，每一个预测结果包含nc+5个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 3个输出层最后的1乘1卷积
       # inplace: 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    # 如果模型不训练那么将会对这些预测得到的参数进一步处理,然后输出,可以方便后期的直接调用
	# 包含了三个信息pred_box [x,y,w,h] pred_conf[confidence] pre_cls[cls0,cls1,cls2,...clsn]
```

这段代码主要是获取预测得到的各种信息。

detection layer 相当于yolov3中的YOLOLayer层，我们解释一下包含的参数：

 *  nc: 分类数量
 *  no:每个anchor的输出数，为(x,y,w,h,conf) + nc = 5 + nc 的总数
 *  nl: 预测层数，此次为3
 *  na:  anchors的数量，此次为3
 *  grid:  格子坐标系，左上角为(1,1),右下角为(input.w/stride,input.h/stride)

### 3.2 向前传播 

```java
'''===================2.向前传播============================'''
    def forward(self, x):
        z = []  # inference output
      
        for i in range(self.nl):

            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
           # 维度重排列: bs, 先验框组数, 检测框行数, 检测框列数, 属性数 + 分类数
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # contiguous 将数据保证内存中位置连续
            '''
            向前传播时需要将相对坐标转换到grid绝对坐标系中
            '''
            if not self.training:  # inference
                '''
                生成坐标系
                grid[i].shape = [1,1,ny,nx,2]
                                [[[[1,1],[1,2],...[1,nx]],
                                [[2,1],[2,2],...[2,nx]],
                                ...,
                                [[ny,1],[ny,2],...[ny,nx]]]]
                '''
                # 换输入后重新设定锚框
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 加载网格点坐标 先验框尺寸
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                '''
                按损失函数的回归方式来转换坐标
                '''
                y = x[i].sigmoid()
                # 改变原数据 计算定位参数
                if self.inplace:
                    # grid: 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # anchor_grid: 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # stride: 是一个grid cell的实际尺寸
                    # 经过sigmoid, 值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5）
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                    # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # 存储每个特征图检测框的信息
                z.append(y.view(bs, -1, self.no))
        # 训练阶段直接返回x
        # 预测阶段返回3个特征图拼接的结果
        return x if self.training else (torch.cat(z, 1), x)
```

这段代码主要是对三个feature map分别进行处理：(n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)

首先进行for循环，每次i的循环，产生一个z。维度重排列：(n, 255, \_, \_) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)，三层分别预测了80\*80、40\*40、20\*20次。

接着 构造网格，因为推理返回的不是归一化后的网格偏移量，需要再加上网格的位置，得到最终的推理坐标，再送入nms。所以这里构建网格就是为了纪律每个grid的网格坐标 方面后面使用

最后按损失函数的回归方式来转换坐标，利用sigmoid激活函数计算定位参数，cat(dim=-1)为直接拼接。注意： 训练阶段直接返回x ，而预测阶段返回3个特征图拼接的结果 

### 3.3 相对坐标转换到grid绝对坐标系 

```java
'''===================3.相对坐标转换到grid绝对坐标系============================'''
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # anchor_grid即每个格子对应的anchor宽高，stride是下采样率，三层分别是8，16，32
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
```

这段代码主要是将相对坐标转换到grid绝对坐标系。

首先构造网格标尺坐标

 *   indexing='ij' ： 表示的是i是同一行，j表示同一列
 *   indexing='xy' ： 表示的是x是同一列，y表示同一行

grid复制成3倍，因为是3个框。anchor\_grid是每个anchor宽高。anchor\_grid = (self.anchors\[i\].clone() \* self.stride\[i\])。注意这里为啥要乘呢？因为在外面已经把anchors给除了对应的下采样率，这里再乘回来。

## 🚀四、Model类 

Model类是整个模型的构造模块部分。 通过自定义YOLO模型类 ，继承torch.nn.Module。主要作用是指定模型的yaml文件以及一系列的训练参数。

### 4.1 \_\_init\_\_函数 

（1） 首先加载yaml文件

```java
class Model(nn.Module):
    '''===================1.__init__函数==========================='''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
     
        # 父类的构造方法
        super().__init__()
        # 检查传入的参数格式，如果cfg是加载好的字典结果
        if isinstance(cfg, dict):
            # 直接保存到模型中
            self.yaml = cfg  # model dict
        # 若不是字典 则为yaml文件路径
        else:  # is *.yaml 一般执行这里
            # 导入yaml文件
            import yaml  # for torch hub
            # 保存文件名：cfg file name = yolov5s.yaml
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 将yaml文件加载为字典
                self.yaml = yaml.safe_load(f)  # model dict 取到配置文件中每条的信息（没有注释内容）
```

\_\_init\_\_函数主要功能为从yaml中初始化model中各种变量包括网络模型model、通道ch，分类数nc，锚框anchor和三层输出缩放倍数stride等。

 *  cfg: YOLO v5模型配置文件 这里使用yolov5s模型
 *  ch:  输入图片的通道数 默认为3
 *  nc:  数据集的类别个数
 *  anchors:  表示anchor框, 一般是None

然后判断输入的cfg是否为字典。一般都不是字典，直接进入else，打开yaml文件，转换成字典格式。

（2）接着开始搭建模型

```java
# Define model
        # 搭建模型
        # yaml.get('ch', ch)表示若不存在键'ch',则返回值ch
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        # 判断类的通道数和yaml中的通道数是否相等，一般不执行，因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            # 在终端给出提示
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 将yaml中的值修改为构造方法中的值
            self.yaml['nc'] = nc  # override yaml value

        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            # 在终端给出提示
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # 将yaml中的值改为构造方法中的值
            self.yaml['anchors'] = round(anchors)  # override yaml value
```

转换为字典以后，就可以进行模型搭建了。首先要判断：

 *  判断yaml中是否有设置输入图像的通道ch，如果没有就默认为3。
 *  判断分类总数nc和anchor与yaml中是否一致，如果不一致，则使用init函数参数中指定的nc和anchor。

（3） 读取yaml中的网络结构并实例化

```java
# 解析模型，self.model是解析后的模型 self.save是每一层与之相连的层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # deepcopy()复杂产生一个新的对象
        # 加载每一类的类别名
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # inplace指的是原地操作 如x+=1 有利于节约内存
        # self.inplace=True  默认True  不使用加速推理
        self.inplace = self.yaml.get('inplace', True)
```

这里是通过parse\_model来进行解析和建立模型的。

到这一步为止，我们的yolo模型的网络架构就已经全部搭建完成了。

（4）最后来计算图像从输入到输出的缩放倍数和anchor在head上的大小 

```java
# Build strides, anchors
        # 构造步长、先验框
        m = self.model[-1]  # Detect()
        # 判断最后一层是否为Detect层
        if isinstance(m, Detect):
            # 定义一个256 * 256大小的输入
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 保存特征层的stride,并且将anchor处理成相对于特征层的格式
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 原始定义的anchor是原始图片上的像素值，要将其缩放至特征图的大小
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            # 将步长保存至模型
            self.stride = m.stride
            # 初始化bias
            self._initialize_biases()  # only run once

        # Init weights, biases
        # 初始化权重
        initialize_weights(self)
        # 打印模型信息
        self.info()
        LOGGER.info('')
```

主要步骤：

1.获取结构最后一层Detect层

2.定义一个256\*256大小的输入

3.将\[1, ch, 256, 256\]大小的tensor进行一次向前传播，得到3层的输出，用输入大小256分别除以输出大小得到每一层的下采样倍数stride  
4.分别用最初的anchor大小除以stride将anchor线性缩放到对应层上

### 4.2 数据增强相关函数 

#### 4.2.1 forward():管理前向传播函数 

```java
# ===1.forward():管理前向传播函数=== #
    def forward(self, x, augment=False, profile=False, visualize=False):
        # 是否在测试时也使用数据增强
        if augment:
            # 增强训练，对数据采取了一些了操作
            return self._forward_augment(x)  # augmented inference, None
        # 默认执行，正常前向推理
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
```

这里做了一个分支，是否在测试时使用数据增强：

 *  如果使用增强训练，则返回推理的forward（\_forward\_augment）
 *  如果不使用增强训练，则返回训练的forward（\_forward\_once）

包含参数：

 *  x：原图
 *  augment ： 是否使用增强式推导
 *  profile：是否测试每个网络层的性能
 *  visualize：是否输出每个网络层的特征图

#### 4.2.2 \_forward\_augment():推理的forward 

```java
# ===2._forward_augment():推理的forward=== #
    # 将图片进行裁剪,并分别送入模型进行检测
    def _forward_augment(self, x):
        # 获得图像的高和宽
        img_size = x.shape[-2:]  # height, width
        # s是规模
        s = [1, 0.83, 0.67]  # scales
        # flip是翻转，这里的参数表示沿着哪个轴翻转
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img函数的作用就是根据传入的参数缩放和翻转图像
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 模型前向传播
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            #  恢复数据增强前的模样
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对不同尺寸进行不同程度的筛选
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
```

这里的代码是在推理的时候做数据增强TTA（Test Time Augmentation），其 x 参数是图像所对应的 tensor。

这个函数只在 val、detect 主函数中使用，用于提高推导的精度。

> 设分类数为 80 、检测框属性数为 5，则基本步骤是：
> 
>  *  对图像进行变换：总共 3 次，分别是 \[ 原图 \]，\[ 尺寸缩小到原来的 0.83，同时水平翻转 \]，\[ 尺寸缩小到原来的 0.67 \]
>  *  对图像使用 \_forward\_once 函数，得到在 eval 模式下网络模型的推导结果。对原图是 shape 为 \[1, 22743, 85\] 的图像检测框信息 (见 Detect 对象的 forward 函数)
>  *  根据 尺寸缩小倍数、翻转维度 对检测框信息进行逆变换，添加进列表 y
>  *  截取 y\[0\] 对大物体的检测结果，保留 y\[1\] 所有的检测结果，截取 y\[2\] 对小物体的检测结果，拼接得到新的检测框信息

#### 4.2.3 \_forward\_once():训练的forward 

```java
# ===3._forward_once():训练的forward=== #
    def _forward_once(self, x, profile=False, visualize=False):
       
        # 各网络层输出, 各网络层推导耗时
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 遍历model的各个模块
        for m in self.model:
            # m.f 就是该层的输入来源，如果不为-1那就不是从上一层而来
            if m.f != -1:  # if not from previous layer
                # from 参数指向的网络层输出的列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 测试该网络层的性能
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用该网络层进行推导, 得到该网络层的输出
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 将每一层的输出结果保存到y
            if visualize:
                # 绘制该 batch 中第一张图像的特征图
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
```

这个函数是训练的forward，对模型每一层进行迭代。

参数为：

 *  x： 输入图像
 *  profile： 是否测试每个网络层的性能，是则调用 self.\_profile\_one\_layer函数
 *  visualize： 是否输出每个网络层的特征图，是则调用 utils.plots.feature\_visualization。这个函数是取 batch 中的第一张图像，然后把每个通道上的二维矩阵看成一张灰度图，分别绘制

#### 4.2.4 \_descale\_pred():将推理结果恢复到原图尺寸 

```java
# ===4._descale_pred():将推理结果恢复到原图尺寸(逆操作)=== #
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
  
        if self.inplace:
            # 把x,y,w,h恢复成原来的大小
            p[..., :4] /= scale  # de-scale
            # bs c h w  当flips=2是对h进行变换，那就是上下进行翻转
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            # 同理flips=3是对水平进行翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
```

这段代码主要是用在上面的\_\_init\_\_函数上，将推理结果恢复到原图图片尺寸TTA中用到

参数为：

 *  p:  推理结果
 *  flips: 翻转标记(2-ud上下, 3-lr左右)
 *  scale:  图片缩放比例
 *  img\_size: 原图图片尺寸

#### 4.2.5 \_clip\_augmented（）:TTA的时候对原图片进行裁剪 

```java
# ===5._clip_augmented（）:TTA的时候对原图片进行裁剪=== #
    # 也是一种数据增强方式，用在TTA测试的时候
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y
```

这个没啥要讲的，其实也是一种数据增强方式，用在TTA测试的时候

#### 4.2.6 \_profile\_one\_layer（）:打印日志信息 

```java
# ===6._profile_one_layer（）:打印日志信息=== #
    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
```

这个函数用于测试每个网络层的性能

参数为：

 *  m： 网络层
 *  x： 该网络层的 from 列表中的网络层输出
 *  dt： 各网络层推导耗时 (列表)

使用 logging 模块输出有：

 *  time (ms)： 前向推导时间
 *  GFLOPs： 浮点运算量，需要安装 thop 模块
 *  params： 网络层参数量
 *  module： 网络层名称

#### 4.2.7 \_initialize\_biases（）:初始化偏置biases信息 

```java
# ===7._initialize_biases（）:初始化偏置biases信息=== #
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
```

这段代码是用来初始化detect的偏置，用在上面的\_\_init\_\_函数上

#### 4.2.8 \_print\_biases（）:打印偏置biases信息 

```java
# ===8._print_biases（）:打印偏置biases信息=== #
    def _print_biases(self):
        """
        打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
```

这段代码主要是打印模型中最后Detect层的偏置bias信息，也可以任选哪些层bias信息

#### 4.2.9 fuse（）:将Conv2d+BN进行融合 

```java
# ===9.fuse（）:将Conv2d+BN进行融合=== #
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
  
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 移除bn
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播
                m.forward = m.forward_fuse  # update forward
        # 打印conv+bn融合后的模型信息
        self.info()
        return self
```

这段代码主要是调用了 fuse\_conv\_and\_bn 这个函数将Conv2d+BN进行融合

用在detect.py、val.py，调用torch\_utils.py中的fuse\_conv\_and\_bn函数和common.py中Conv模块的fuseforward函数。最后打印出融合后的模型信息。

#### 4.2.10 autoshape（）:扩展模型功能 

```java
# ===10.autoshape（）:扩展模型功能=== #
    def autoshape(self):  # add AutoShape module
      
        LOGGER.info('Adding AutoShape... ')
        #  此时模型包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m
```

这段代码直接调用common.py中的AutoShape模块，也是一个扩展模型功能的模块，来实现扩展模型功能

#### 4.2.11 info():打印模型结构信息 

```java
# ===11.info():打印模型结构信息=== #
    def info(self, verbose=False, img_size=640):  # print model information
     
        model_info(self, verbose, img_size)
```

这段代码是用在上面的\_\_init\_\_函数上调用torch\_utils.py下model\_info函数打印模型信息

#### 4.2.12 \_apply():将模块转移到 CPU/ GPU上 

```java
# ===12._apply():将模块转移到 CPU/ GPU上=== #
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self
```

这段代码主要是将模块转移到 CPU/ GPU上

pytorch中的model.apply(fn)会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身。经常用于初始化init\_weights的操作。

（终于写完这一堆函数了，寝室熄灯太难了555）

这里面除了模型搭建还包含很多拓展功能，我们主要看的是\_\_init\_\_和\_\_forward\_\_两个函数。

## 🚀五、yolo.py全部注释 

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

'''===============================================一、导入包==================================================='''
'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
from copy import deepcopy  # 数据拷贝模块 深拷贝
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve() # __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/modles/yolo.py
ROOT = FILE.parents[1]  # YOLOv5 root directory 保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
# ROOT = ROOT.relative_to(Path.cwd())  # relative  ROOT设置为相对路径

'''===================3..加载自定义模块============================'''
from models.common import * # yolov5的网络结构(yolov5)
from models.experimental import *   # 导入在线下载模块
from utils.autoanchor import check_anchor_order   # 导入检查anchors合法性的函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args   # 定义了一些常用的工具函数
from utils.plots import feature_visualization  # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)   # 定义了一些与PyTorch有关的工具函数

# 导入thop包 用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

'''===============================================二、Detect模块==================================================='''
'''
   Detect模块是用来构建Detect层的，将输入feature map 通过一个卷积操作和公式计算到我们想要的shape, 为后面的计算损失或者NMS后处理作准备
'''
class Detect(nn.Module):
    stride = None  # 特征图的缩放步长
    onnx_dynamic = False  # ONNX动态量化

    '''===================1.获取预测得到的参数============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """
        detection layer 相当于yolov3中的YOLOLayer层
          :params nc: 数据集类别数量
          :params anchors: 传入3个feature map上的所有anchor的大小（P3、P4、P5）
          :params ch: [128, 256, 512] 3个输出feature map的channel
          """
        super().__init__()

        # nc: 数据集类别数量
        self.nc = nc
        # no: 表示每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # nc+5=nc+(x,y,w,h,conf)
        # nl: 表示预测层数，yolov5是3层预测
        self.nl = len(anchors)
        # na: 表示anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个anchor
        self.na = len(anchors[0]) // 2
        # grid: 表示初始化grid列表大小，下面会计算grid，grid就是每个格子的x，y坐标（整数，比如0-19），左上角为(1,1),右下角为(input.w/stride,input.h/stride)
        self.grid = [torch.zeros(1)] * self.nl
        # anchor_grid: 表示初始化anchor_grid列表大小，空列表
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        # 注册常量anchor，并将预选框（尺寸）以数对形式存入，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2) 注意后面就可以通过self.anchors来访问它了
        # 每一张进行三次预测，每一个预测结果包含nc+5个值
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20) --> ch=(255, 255, 255)
        # 255 -> (nc+5)*3 ===> 为了提取出预测框的位置信息以及预测框尺寸信息
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 3个输出层最后的1乘1卷积
       # inplace: 一般都是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    # 如果模型不训练那么将会对这些预测得到的参数进一步处理,然后输出,可以方便后期的直接调用
	# 包含了三个信息pred_box [x,y,w,h] pred_conf[confidence] pre_cls[cls0,cls1,cls2,...clsn]

    '''===================2.向前传播============================'''
    def forward(self, x):
        z = []  # inference output
        # 对三个feature map分别进行处理
        # (n, 255, 80, 80),(n, 255, 40, 40),(n, 255, 20, 20)
        for i in range(self.nl):

            # 下面3行代码的工作：
            # (n, 255, _, _) -> (n, 3, nc+5, ny, nx) -> (n, 3, ny, nx, nc+5)
            # 相当于三层分别预测了80*80、40*40、20*20次，每一次预测都包含3个框

            # 进行1*1卷积，经过这个卷积就变成（5+分类数）个通道了
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
           # 维度重排列: bs, 先验框组数, 检测框行数, 检测框列数, 属性数 + 分类数
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # contiguous 将数据保证内存中位置连续

            '''
            向前传播时需要将相对坐标转换到grid绝对坐标系中
            '''
            if not self.training:  # inference
                '''
                生成坐标系
                grid[i].shape = [1,1,ny,nx,2]
                                [[[[1,1],[1,2],...[1,nx]],
                                [[2,1],[2,2],...[2,nx]],
                                ...,
                                [[ny,1],[ny,2],...[ny,nx]]]]
                '''
                # 换输入后重新设定锚框
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # 加载网格点坐标 先验框尺寸
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                '''
                按损失函数的回归方式来转换坐标
                '''
                y = x[i].sigmoid()
                # 改变原数据 计算定位参数
                if self.inplace:
                    # grid: 位置基准 或者理解为 cell的预测初始位置，而y[..., 0:2]是作为在grid坐标基础上的位置偏移
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # anchor_grid: 预测框基准 或者理解为 预测框的初始位置，而 y[..., 2:4]是作为预测框位置的调整
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # stride: 是一个grid cell的实际尺寸
                    # 经过sigmoid, 值范围变成了(0-1),下一行代码将值变成范围（-0.5，1.5）
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    # 范围变成(0-4)倍，设置为4倍的原因是下层的感受野是上层的2倍
                    # 因下层注重检测大目标，相对比上层而言，计算量更小，4倍是一个折中的选择
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # 存储每个特征图检测框的信息
                z.append(y.view(bs, -1, self.no))
        # 训练阶段直接返回x
        # 预测阶段返回3个特征图拼接的结果
        return x if self.training else (torch.cat(z, 1), x)

    '''===================3.相对坐标转换到grid绝对坐标系============================'''
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            # 网格标尺坐标
            # indexing='ij' 表示的是i是同一行，j表示同一列
            # indexing='xy' 表示的是x是同一列，y表示同一行
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        # grid --> (20, 20, 2), 复制成3倍，因为是三个框 -> (3, 20, 20, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # anchor_grid即每个格子对应的anchor宽高，stride是下采样率，三层分别是8，16，32，这里为啥要乘呢，因为在外面已经把anchors给除了对应的下采样率，这里再乘回来
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

'''===============================================三、Model模块==================================================='''
class Model(nn.Module):
    '''===================1.__init__函数==========================='''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
           :params cfg:YOLO v5模型配置文件 这里使用yolov5s模型
           :params ch: 输入图片的通道数 默认为3
           :params nc: 数据集的类别个数
           :anchors: 表示anchor框, 一般是None
        """
        # 父类的构造方法
        super().__init__()
        # 检查传入的参数格式，如果cfg是加载好的字典结果
        if isinstance(cfg, dict):
            # 直接保存到模型中
            self.yaml = cfg  # model dict
        # 若不是字典 则为yaml文件路径
        else:  # is *.yaml 一般执行这里
            # 导入yaml文件
            import yaml  # for torch hub
            # 保存文件名：cfg file name = yolov5s.yaml
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                # 将yaml文件加载为字典
                self.yaml = yaml.safe_load(f)  # model dict 取到配置文件中每条的信息（没有注释内容）

        '''===================2.获取输入通道============================'''
        # Define model
        # 搭建模型
        # yaml.get('ch', ch)表示若不存在键'ch',则返回值ch
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        # 判断类的通道数和yaml中的通道数是否相等，一般不执行，因为nc=self.yaml['nc']恒成立
        if nc and nc != self.yaml['nc']:
            # 在终端给出提示
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # 将yaml中的值修改为构造方法中的值
            self.yaml['nc'] = nc  # override yaml value

        # 重写anchor，一般不执行, 因为传进来的anchors一般都是None
        if anchors:
            # 在终端给出提示
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # 将yaml中的值改为构造方法中的值
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 解析模型，self.model是解析后的模型 self.save是每一层与之相连的层
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # deepcopy()复杂产生一个新的对象
        # 加载每一类的类别名
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # inplace指的是原地操作 如x+=1 有利于节约内存
        # self.inplace=True  默认True  不使用加速推理
        self.inplace = self.yaml.get('inplace', True)

        '''===================3.获取Detect输出模块============================'''
        # Build strides, anchors
        # 构造步长、先验框
        m = self.model[-1]  # Detect()
        # 判断最后一层是否为Detect层
        if isinstance(m, Detect):
            # 定义一个256 * 256大小的输入
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # 保存特征层的stride,并且将anchor处理成相对于特征层的格式
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 原始定义的anchor是原始图片上的像素值，要将其缩放至特征图的大小
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序与stride顺序是否一致
            check_anchor_order(m)
            # 将步长保存至模型
            self.stride = m.stride
            # 初始化bias
            self._initialize_biases()  # only run once

        # Init weights, biases
        # 初始化权重
        initialize_weights(self)
        # 打印模型信息
        self.info()
        LOGGER.info('')

    '''===================4.数据增强============================'''

    # ===1.forward():管理前向传播函数=== #
    def forward(self, x, augment=False, profile=False, visualize=False):
        # 是否在测试时也使用数据增强
        if augment:
            # 增强训练，对数据采取了一些了操作
            return self._forward_augment(x)  # augmented inference, None
        # 默认执行，正常前向推理
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # ===2._forward_augment():推理的forward=== #
    # 将图片进行裁剪,并分别送入模型进行检测
    def _forward_augment(self, x):
        # 获得图像的高和宽
        img_size = x.shape[-2:]  # height, width
        # s是规模
        s = [1, 0.83, 0.67]  # scales
        # flip是翻转，这里的参数表示沿着哪个轴翻转
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img函数的作用就是根据传入的参数缩放和翻转图像
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 模型前向传播
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            #  恢复数据增强前的模样
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        # 对不同尺寸进行不同程度的筛选
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    # ===3._forward_once():训练的forward=== #
    def _forward_once(self, x, profile=False, visualize=False):
        """
        :params x: 输入图像
        :params profile: True 可以做一些性能评估
        :params feature_vis: True 可以做一些特征可视化
        :return train: 一个tensor list 存放三个元素   [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                        分别是 [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
                 inference: 0 [1, 19200+4800+1200, 25] = [bs, anchor_num*grid_w*grid_h, xywh+c+20classes]
                                   1 一个tensor list 存放三个元素 [bs, anchor_num, grid_w, grid_h, xywh+c+20classes]
                                     [1, 3, 80, 80, 25] [1, 3, 40, 40, 25] [1, 3, 20, 20, 25]
        """
        # 各网络层输出, 各网络层推导耗时
        # y: 存放着self.save=True的每一层的输出，因为后面的层结构concat等操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 遍历model的各个模块
        for m in self.model:
            # m.f 就是该层的输入来源，如果不为-1那就不是从上一层而来
            if m.f != -1:  # if not from previous layer
                # from 参数指向的网络层输出的列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 测试该网络层的性能
            if profile:
                self._profile_one_layer(m, x, dt)
            # 使用该网络层进行推导, 得到该网络层的输出
            x = m(x)  # run
            # 存放着self.save的每一层的输出，因为后面需要用来作concat等操作要用到  不在self.save层的输出就为None
            y.append(x if m.i in self.save else None)  # save output
            # 将每一层的输出结果保存到y
            if visualize:
                # 绘制该 batch 中第一张图像的特征图
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # ===4._descale_pred():将推理结果恢复到原图尺寸(逆操作)=== #
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        ''' 用在上面的__init__函数上
           将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
           :params p: 推理结果
           :params flips: 翻转标记(2-ud上下, 3-lr左右)
           :params scale: 图片缩放比例
           :params img_size: 原图图片尺寸
        '''
        if self.inplace:
            # 把x,y,w,h恢复成原来的大小
            p[..., :4] /= scale  # de-scale
            # bs c h w  当flips=2是对h进行变换，那就是上下进行翻转
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            # 同理flips=3是对水平进行翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    # ===5._clip_augmented（）:TTA的时候对原图片进行裁剪=== #
    # 也是一种数据增强方式，用在TTA测试的时候
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    # ===6._profile_one_layer（）:打印日志信息=== #
    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # ===7._initialize_biases（）:初始化偏置biases信息=== #
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # ===8._print_biases（）:打印偏置biases信息=== #
    def _print_biases(self):
        """
        打印模型中最后Detect层的偏置bias信息(也可以任选哪些层bias信息)
        """
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    # ===9.fuse（）:将Conv2d+BN进行融合=== #
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用oneflow_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构, 那么就调用fuse_conv_and_bn函数讲conv和bn进行融合, 加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 更新卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # 移除bn
                delattr(m, 'bn')  # remove batchnorm
                # 更新前向传播
                m.forward = m.forward_fuse  # update forward
        # 打印conv+bn融合后的模型信息
        self.info()
        return self

    # ===10.autoshape（）:扩展模型功能=== #
    def autoshape(self):  # add AutoShape module
        """
        add AutoShape module  直接调用common.py中的AutoShape模块  也是一个扩展模型功能的模块
        """
        LOGGER.info('Adding AutoShape... ')
        #  此时模型包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    # ===11.info():打印模型结构信息=== #
    def info(self, verbose=False, img_size=640):  # print model information
        """
        用在上面的__init__函数上调用torch_utils.py下model_info函数打印模型信息
        """
        model_info(self, verbose, img_size)

    # ===12._apply():将模块转移到 CPU/ GPU上=== #
    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

'''===============================================四、parse_model模块==================================================='''
def parse_model(d, ch):  # model_dict, input_channels(3)

    '''===================1. 获取对应参数============================'''

    # 使用 logging 模块输出列标签
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 获取anchors，nc，depth_multiple，width_multiple，这些参数在介绍yaml时已经介绍过
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    # na: 每组先验框包含的先验框数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no: na * 属性数 (5 + 分类数)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    '''===================2. 开始搭建网络============================'''

    # 网络单元列表, 网络输出引用列表, 当前的输出通道数
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 读取 backbone, head 中的网络单元
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 利用 eval 函数, 读取 model 参数对应的类名
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 使用 eval 函数将字符串转换为变量
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        '''===================3. 更新当前层的参数，计算c2============================'''
        # depth gain: 控制深度，如yolov5s: n*0.33，n: 当前模块的次数(间接控制深度)
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 当该网络单元的参数含有: 输入通道数, 输出通道数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            # c1: 当前层的输入channel数; c2: 当前层的输出channel数(初定); ch: 记录着所有层的输出channel数
            c1, c2 = ch[f], args[0]
            # no=75，只有最后一层c2=no，最后一层不用控制宽度，输出channel必须是no
            if c2 != no:  # if not output
                # width gain: 控制宽度，如yolov5s: c2*0.5; c2: 当前层的最终输出channel数(间接控制宽度)
                c2 = make_divisible(c2 * gw, 8)

            '''===================4.使用当前层的参数搭建当前层============================'''
            # 在初始args的基础上更新，加入当前层的输入channel并更新当前层
            # [in_channels, out_channels, *args[1:]]
            args = [c1, c2, *args[1:]]
            # 如果当前层是BottleneckCSP/C3/C3TR/C3Ghost/C3x，则需要在args中加入Bottleneck的个数
            # [in_channels, out_channels, Bottleneck个数, Bool(shortcut有无标记)]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                # 在第二个位置插入bottleneck个数n
                args.insert(2, n)  # number of repeats
                # 恢复默认值1
                n = 1
        # 判断是否是归一化模块
        elif m is nn.BatchNorm2d:
            # BN层只需要返回上一层的输出channel
            args = [ch[f]]
        # 判断是否是tensor连接模块
        elif m is Concat:
            # Concat层则将f中所有的输出累加得到这层的输出channel
            c2 = sum(ch[x] for x in f)
        # 判断是否是detect模块
        elif m is Detect:
            # 在args中加入三个Detect层的输出channel
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors 几乎不执行
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract: # 不怎么用
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:  # 不怎么用
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]  # args不变

        '''===================5.打印和保存layers信息============================'''
        # m_: 得到当前层的module，将n个模块组合存放到m_里面
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module

        # 打印当前层结构的一些基本信息
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算这一层的参数量
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print

        # 把所有层结构中的from不是-1的值记下 [6,4,14,10,17,20,23]
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        # 将当前层结构module加入layers中
        layers.append(m_)
        if i == 0:
            ch = [] # 去除输入channel[3]
        # 把当前层的输出channel数加入ch
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # 创建解析器
    # --cfg: 模型配置文件
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    # --device: 选用设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --profile: 用户配置文件
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    # --test: 测试
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()# 增加后的属性赋值给args
    opt.cfg = check_yaml(opt.cfg)  # 检查YAML文件
    print_args(FILE.stem, opt) # 检测YOLO v5的github仓库是否更新,若已更新,给出提示
    device = select_device(opt.device) # 选择设备

    # Create model
    # 构造模型
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    #用户自定义配置
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    # 测试所有的模型
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
```

> 本文参考：
> 
> [【YOLOV5-5.x 源码解读】yolo.py\_yolov5 yolo.py\_满船清梦压星河HK的博客-CSDN博客][YOLOV5-5.x _yolo.py_yolov5 yolo.py_HK_-CSDN]  
> [YOLOv5 源码解析 —— 网络模型建立\_yolov5网络模型\_荷碧TongZJ的博客][YOLOv5 _ _ _yolov5_TongZJ]

![](https://i-blog.csdnimg.cn/blog_migrate/41d1f4a701f6626317730f0415ae5865.gif)


[mirrors _ ultralytics _ yolov5 _ GitCode]: https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator
[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764
[Link 1]: #%E5%89%8D%E8%A8%80
[Link 2]: #%F0%9F%9A%80%E4%B8%80%E3%80%81%20%E5%AF%BC%E5%8C%85%E5%92%8C%E5%9F%BA%E6%9C%AC%E9%85%8D%E7%BD%AE
[1.1 _python_]: #1.1%20%E5%AF%BC%E5%85%A5%E5%AE%89%E8%A3%85%E5%A5%BD%E7%9A%84python%E5%BA%93%C2%A0
[1.2]: #1.2%20%E8%8E%B7%E5%8F%96%E5%BD%93%E5%89%8D%E6%96%87%E4%BB%B6%E7%9A%84%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84
[1.3]: #1.4%20%E5%8A%A0%E8%BD%BD%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97
[parse_model]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81parse_model%E5%87%BD%E6%95%B0
[2.1]: #2.1%20%E8%8E%B7%E5%8F%96%E5%AF%B9%E5%BA%94%E5%8F%82%E6%95%B0
[2.2]: #%C2%A02.2%20%E6%90%AD%E5%BB%BA%E7%BD%91%E7%BB%9C%E5%89%8D%E5%87%86%E5%A4%87
[2.3 _c2]: #2.3%20%E6%9B%B4%E6%96%B0%E5%BD%93%E5%89%8D%E5%B1%82%E7%9A%84%E5%8F%82%E6%95%B0%EF%BC%8C%E8%AE%A1%E7%AE%97c2
[2.4]: #2.4%20%E4%BD%BF%E7%94%A8%E5%BD%93%E5%89%8D%E5%B1%82%E7%9A%84%E5%8F%82%E6%95%B0%E6%90%AD%E5%BB%BA%E5%BD%93%E5%89%8D%E5%B1%82
[2.5 _layers]: #%C2%A02.5%20%E6%89%93%E5%8D%B0%E5%92%8C%E4%BF%9D%E5%AD%98layers%C2%A0
[_Detect]: #%F0%9F%9A%80%C2%A0%E4%B8%89%E3%80%81Detect%E6%A8%A1%E5%9D%97
[3.1]: #%C2%A03.1%20%E8%8E%B7%E5%8F%96%E9%A2%84%E6%B5%8B%E5%BE%97%E5%88%B0%E7%9A%84%E5%8F%82%E6%95%B0
[3.2]: #3.2%20%E5%90%91%E5%89%8D%E4%BC%A0%E6%92%AD
[3.3 _grid]: #3.3%20%E7%9B%B8%E5%AF%B9%E5%9D%90%E6%A0%87%E8%BD%AC%E6%8D%A2%E5%88%B0grid%E7%BB%9D%E5%AF%B9%E5%9D%90%E6%A0%87%E7%B3%BB
[Model]: #%F0%9F%9A%80%E5%9B%9B%E3%80%81Model%E7%B1%BB
[4.1 _init]: #4.1%C2%A0__init__%E5%87%BD%E6%95%B0
[4.2]: #4.2%20%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E7%9B%B8%E5%85%B3%E5%87%BD%E6%95%B0
[4.2.1 forward]: #4.2.1%C2%A0forward%28%29%3A%E7%AE%A1%E7%90%86%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%87%BD%E6%95%B0
[4.2.2 _forward_augment_forward]: #4.2.2%C2%A0_forward_augment%28%29%3A%E6%8E%A8%E7%90%86%E7%9A%84forward
[4.2.3 _forward_once_forward]: #%C2%A04.2.3%C2%A0_forward_once%28%29%3A%E8%AE%AD%E7%BB%83%E7%9A%84forward
[4.2.4 _descale_pred]: #%C2%A04.2.4%C2%A0_descale_pred%28%29%3A%E5%B0%86%E6%8E%A8%E7%90%86%E7%BB%93%E6%9E%9C%E6%81%A2%E5%A4%8D%E5%88%B0%E5%8E%9F%E5%9B%BE%E5%B0%BA%E5%AF%B8
[4.2.5 _clip_augmented_TTA]: #4.2.5%C2%A0_clip_augmented%EF%BC%88%EF%BC%89%3ATTA%E7%9A%84%E6%97%B6%E5%80%99%E5%AF%B9%E5%8E%9F%E5%9B%BE%E7%89%87%E8%BF%9B%E8%A1%8C%E8%A3%81%E5%89%AA
[4.2.6 _profile_one_layer]: #%C2%A04.2.6%C2%A0_profile_one_layer%EF%BC%88%EF%BC%89%3A%E6%89%93%E5%8D%B0%E6%97%A5%E5%BF%97%E4%BF%A1%E6%81%AF
[4.2.7 _initialize_biases_biases]: #%C2%A04.2.7%C2%A0_initialize_biases%EF%BC%88%EF%BC%89%3A%E5%88%9D%E5%A7%8B%E5%8C%96%E5%81%8F%E7%BD%AEbiases%E4%BF%A1%E6%81%AF
[4.2.8 _print_biases_biases]: #4.2.8%C2%A0_print_biases%EF%BC%88%EF%BC%89%3A%E6%89%93%E5%8D%B0%E5%81%8F%E7%BD%AEbiases%E4%BF%A1%E6%81%AF
[4.2.9 fuse_Conv2d_BN]: #4.2.9%C2%A0fuse%EF%BC%88%EF%BC%89%3A%E5%B0%86Conv2d%2BBN%E8%BF%9B%E8%A1%8C%E8%9E%8D%E5%90%88
[4.2.10 autoshape]: #4.2.10%20autoshape%EF%BC%88%EF%BC%89%3A%E6%89%A9%E5%B1%95%E6%A8%A1%E5%9E%8B%E5%8A%9F%E8%83%BD
[4.2.11 info]: #4.2.11%20info%28%29%3A%E6%89%93%E5%8D%B0%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%BF%A1%E6%81%AF
[4.2.12 _apply_ CPU_ GPU]: #4.2.12%20_apply%28%29%3A%E5%B0%86%E6%A8%A1%E5%9D%97%E8%BD%AC%E7%A7%BB%E5%88%B0%20CPU%2F%20GPU%E4%B8%8A
[yolo.py]: #%F0%9F%9A%80%E4%BA%94%E3%80%81yolo.py%E5%85%A8%E9%83%A8%E6%B3%A8%E9%87%8A
[YOLOV5-5.x _yolo.py_yolov5 yolo.py_HK_-CSDN]: https://hukai.blog.csdn.net/article/details/119869762
[YOLOv5 _ _ _yolov5_TongZJ]: https://blog.csdn.net/qq_55745968/article/details/124512331?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167961706616800188538901%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167961706616800188538901&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-124512331-null-null.142%5Ev76%5Epc_search_v2,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=strides%20computed%20during%20build&spm=1018.2226.3001.4187