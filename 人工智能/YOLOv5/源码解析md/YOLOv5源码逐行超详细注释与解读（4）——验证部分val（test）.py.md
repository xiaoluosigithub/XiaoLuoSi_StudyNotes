![](https://i-blog.csdnimg.cn/blog_migrate/5b8e236d181b2ec4f964d93b7b0bd0ff.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/e0136aa6ce05568641486f2d7ba52658.jpeg)

## 前言 

本篇文章主要是对YOLOv5项目的验证部分。这个文件之前是叫test.py，后来改为val.py。

在之前我们已经学习了推理部分detect.py和训练部分train.py这两个，而我们今天要介绍的验证部分val.py这个文件主要是train.py每一轮训练结束后，用val.py去验证当前模型的mAP、混淆矩阵等指标以及各个超参数是否是最佳， 不是最佳的话修改train.py里面的结构；确定是最佳了再用detect.py去泛化使用。

总结一下这三个文件的区别：

 *  detect.py： 推理部分。获取实际中最佳推理结果
 *  train.py： 训练部分。读取数据集，加载模型并训练
 *  val.py：验证部分。获取当前数据集上的最佳验证结果

文章代码逐行手打注释，每个模块都有对应讲解，一文帮你梳理整个代码逻辑！

友情提示：全文近5万字，可以先点![](https://i-blog.csdnimg.cn/blog_migrate/ea5f7225888a49f6a6827b9ae71e856f.gif)再慢慢看哦~

源码下载地址：[mirrors / ultralytics / yolov5 · GitCode][mirrors _ ultralytics _ yolov5 _ GitCode]

![](https://i-blog.csdnimg.cn/blog_migrate/76e7ebb46bddfb9ace8fcd78ba36fed5.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]  
  
[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]  
  
[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

## 目录 

[前言 ][Link 1]

[目录][Link 2]

[🚀一、导包与基本配置][Link 3]

[1.1 导入安装好的python库][1.1 _python]

[1.2 获取当前文件的绝对路径][1.2]

[1.3 加载自定义模块][1.3]

[🚀二、保存信息][Link 4]

[2.1 保存预测信息到txt文件][2.1 _txt]

[2.2 保存预测信息到coco格式的json字典][2.2 _coco_json]

[🚀三、计算指标 ][Link 5]

[🚀四、执行run（）函数][run]

[4.1 设置参数][4.1]

[4.2 初始化/加载模型以及设置设备][4.2]

[4.3 加载配置][4.3]

[4.4 加载val数据集][4.4 _val]

[4.5 初始化][4.5]

[4.6 验证过程][4.6]

[4.6.1 开始验证前的预处理][4.6.1]

[4.6.2 前项推理][4.6.2]

[4.6.3 计算损失][4.6.3]

[4.6.4 NMS获得预测框][4.6.4 NMS]

[4.6.5 统计真实框、预测框信息][4.6.5]

[4.6.6 画出前三个batch图片的gt和pred框][4.6.6 _batch_gt_pred]

[4.6.7 计算指标][4.6.7]

[4.6.8 打印日志 ][4.6.8 _]

[4.6.9 保存验证结果 ][4.6.9 _]

[4.6.10 返回结果][4.6.10]

[🚀五、设置opt参数][opt]

[🚀六、执行main()函数][main]

[🚀七、val.py代码全部注释][val.py]

## ![](https://i-blog.csdnimg.cn/blog_migrate/ef9748e3072ad8e03eea7466a1e5b950.gif) 🚀一、导包与基本配置 

### 1.1 导入安装好的python库 

```java
'''============1.导入安装好的python库=========='''

import argparse # 解析命令行参数的库
import json # 实现字典列表和JSON字符串之间的相互解析
import os  # 与操作系统进行交互的文件库 包含文件路径操作与解析
import sys # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread # python中处理多线程的库

import numpy as np # 矩阵计算基础库
import torch # pytorch 深度学习库
from tqdm import tqdm  # 用于直观显示进度条的一个库
```

首先，导入一下常用的python库：

 *  argparse： 它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
 *  json： 实现字典列表和JSON字符串之间的相互解析
 *  os： 它提供了多种操作系统的接口。通过os模块提供的操作系统接口，我们可以对操作系统里文件、终端、进程等进行操作
 *  sys： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
 *  pathlib： 这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读
 *  threading： python中处理多线程的库

然后再导入一些 pytorch库：

 *  numpy： 科学计算库，提供了矩阵，线性代数，傅立叶变换等等的解决方案, 最常用的是它的N维数组对象
 *  torch：这是主要的Pytorch库。它提供了构建、训练和评估神经网络的工具
 *  tqdm：  就是我们看到的训练时进度条显示

### 1.2 获取当前文件的绝对路径 

```java
'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()# __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/val.py
ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path: # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径
```

这段代码会获取当前文件的绝对路径，并使用Path库将其转换为Path对象。

这一部分的主要作用有两个：

 *  将当前项目添加到系统路径上，以使得项目中的模块可以调用。
 *  将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

### 1.3 加载自定义模块 

```java
'''===================3..加载自定义模块============================'''
from models.common import DetectMultiBackend # yolov5的网络结构(yolov5)
from utils.callbacks import Callbacks # 和日志相关的回调函数
from utils.datasets import create_dataloader # 加载数据集的函数
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)  # 定义了一些常用的工具函数
from utils.metrics import ConfusionMatrix, ap_per_class # 在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权
from utils.plots import output_to_target, plot_images, plot_val_study # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import select_device, time_sync  # 定义了一些与PyTorch有关的工具函数
```

这些都是用户自定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换。具体来说，代码从如下几个文件中导入了部分函数和类：

 *  models.common： yolov5的网络结构(yolov5)
 *  utils.callbacks： 定义了回调函数，主要为logger服务
 *  utils.datasets： dateset和dateloader定义代码
 *  utils.general.py： 定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
 *  utils.metrics： 模型验证指标，包括ap，混淆矩阵等
 *  utils.plots.py： 定义了Annotator类，可以在图像上绘制矩形框和标注信息
 *  utils.torch\_utils.py： 定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等 通过导入这些模块，可以更方便地进行目标检测的相关任务，并且减少了代码的复杂度和冗余

## 🚀二、保存信息 

### 2.1 保存预测信息到txt文件 

```java
'''======================1.保存预测信息到txt文件====================='''
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
    for *xyxy, conf, cls in predn.tolist():
        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽高)格式，并归一化，转化为列表再保存
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # line的形式是： "类别 xywh"，若save_conf为true，则line的形式是："类别 xywh 置信度"
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 将上述test得到的信息输出保存 输出为xywh格式 coco数据格式也为xywh格式
        with open(file, 'a') as f:
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
```

这段代码主要是保存预测信息为txt文件

保存的信息为：

 *  cls： 图片类别
 *  xywh： 图片的中心点＋宽高
 *  conf：  置信度

首先获取图片的w和h，也就是对应的宽高，然后把每个图片的预测信息分别存入save\_dir/labels下的xxx.txt中。

接着将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点 + bbox的w,h的格式，并进行归一化。即：xyxy（左上右下） ——> xywh（中心宽高）。

最后，将预测的类别和坐标值保存到对应图片image\_name.txt文件中，路径默认为“runs\\detect\\exp\*\\labels”

### 2.2 保存预测信息到coco格式的json字典 

```java
'''======================2.保存预测信息到coco格式的json字典====================='''
def save_one_json(predn, jdict, path, class_map):
    # 储存格式 {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    # 获取图片id
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 获取预测框 并将xyxy转为xywh格式
    box = xyxy2xywh(predn[:, :4])  # xywh
 
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # 序列解包
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id, # 图片id 即属于哪张图片
                      'category_id': class_map[int(p[5])], # 类别 coco91class()从索引0~79映射到索引0~90
                      'bbox': [round(x, 3) for x in b], # 预测框坐标
                      'score': round(p[4], 5)}) # 预测得分
```

这段代码主要是保存coco格式的json文件字典。

保存的信息为：

 *  image\_id： 图片id，即属于哪张图片
 *  category\_id： 类别，coco91class()从索引0~79映射到索引0~90
 *  bbox：  预测框坐标
 *  score： 预测得分

首先获取图片的id以及预测框，并把xyxy格式转化为xywh格式。

> 注意：之前的的xyxy格式是左上角右下角坐标 ，xywh是中心的坐标和宽高，而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)，所以 box\[:, :2\] -= box\[:, 2:\] / 2 这行代码是将中心点坐标 -> 左上角坐标。

然后再用zip（）函数进行序列解包，逐一保存上述信息。

## 🚀三、计算指标 

```java
'''========================三、计算指标==========================='''
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    返回每个预测框在10个IoU阈值上是TP还是FP
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
  
    # iou超过阈值而且类别正确，则为True，返回索引
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    # 如果存在符合条件的预测框
    if x[0].shape[0]: # 至少有一个TP
        # 将符合条件的位置构建成一个新的矩阵，第一列是行索引（表示gt索引），第二列是列索引（表示预测框索引），第三列是iou值
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            # argsort获得有小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引，对matches矩阵进行排序
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            '''
            参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
            matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
            '''
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            '''
            matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
            '''
            # 以上操作实现了为每一个gt分配一个iou最高的类别的预测框，实现一一对应

        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        '''
         当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
         需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
        '''
        # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
    return correct
```

这段代码主要是计算correct，来获取匹配预测框的iou信息。

这个函数主要有两个作用：

 *  作用1：对预测框与gt进行匹配
 *  作用2：对匹配上的预测框进行iou数值判断，用True来填充，其余没有匹配上的预测框的所以行数全部设置为False

对于每张图像的预测框，需要筛选出能与gt匹配的框来进行相关的iou计算，设置了iou从0.5-0.95的10个梯度，如果匹配的预测框iou大于相对于的阈值，则在对应位置设置为True，否则设置为False；而对于没有匹配上的预测框全部设置为False。

> Q：为什么要筛选？
> 
> 这是因为一个gt只可能是一个类别，不可能是多个类别，所以需要取置信度最高的类别进行匹配。但是此时还可能多个gt和一个预测框匹配，同样的，为这个预测框分配iou值最高的gt，依次来实现一一配对。

## 🚀四、执行run（）函数 

### 4.1 设置参数 

```java
'''======================1.设置参数====================='''
@torch.no_grad()
def run(data, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
        weights=None,  # 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s
        batch_size=32,  # 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
        imgsz=640,  # 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
        conf_thres=0.001,  # object置信度阈值 默认0.001
        iou_thres=0.6,  # 进行NMS时IOU的阈值 默认0.6
        task='val',  # 设置测试的类型 有train, val, test, speed or study几种 默认val
        device='',  # 执行 val.py 所在的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # 数据集是否只有一个类别 默认False
        augment=False,  # 测试时增强
        verbose=False,  # 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
        save_txt=False,  # 是否以txt文件的形式保存模型预测框的坐标 默认True
        save_hybrid=False,  # 是否保存预测每个目标的置信度到预测txt文件中 默认True
        save_conf=False,  # 保存置信度
        save_json=False,  # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）,
                      #运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
        project=ROOT / 'runs/val',  # 验证结果保存的根目录 默认是 runs/val
        name='exp',  # 验证结果保存的目录 默认是exp  最终: runs/val/exp
        exist_ok=False,  # 如果文件存在就increment name，不存在就新建  默认False(默认文件都是不存在的)
        half=True,  # 使用 FP16 的半精度推理
        dnn=False,  # 在 ONNX 推理时使用 OpenCV DNN 后段端
        model=None,  # 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
        dataloader=None, # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
        save_dir=Path(''), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
        plots=True, # 是否可视化 运行val.py传入，默认True
        callbacks=Callbacks(),  # 回调函数
        compute_loss=None, # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
        ):
```

这段代码定义了run（）函数，并设置了一系列参数，用于指定物体检测或识别的相关参数。

这些参数包括：

 *  data： 数据集文件的路径，默认为COCO128数据集的配置文件路径
 *  weights： 模型权重文件的路径，默认为YOLOv5s的权重文件路径
 *  batch\_size: 前向传播的批次大小，运行val.py传入默认32 。运行train.py则传入batch\_size // WORLD\_SIZE \* 2
 *  imgsz： 输入图像的大小，默认为640x640
 *  conf\_thres： 置信度阈值，默认为0.001
 *  iou\_thres： 非极大值抑制的iou阈值，默认为0.6
 *  task:  设置测试的类型 有train, val, test, speed or study几种，默认val
 *  device： 使用的设备类型，默认为空，表示自动选择最合适的设备
 *  single\_cls:  数据集是否只用一个类别，运行val.py传入默认False 运行train.py则传入single\_cls
 *  augment： 是否使用数据增强的方式进行检测，默认为False
 *  verbose:  是否打印出每个类别的mAP，运行val.py传入默认Fasle。运行train.py则传入nc < 50 and final\_epoch
 *  save\_txt： 是否将检测结果保存为文本文件，默认为False
 *  save\_hybrid: 是否保存 label+prediction hybrid results to \*.txt 默认False
 *  save\_conf： 是否在保存的文本文件中包含置信度信息，默认为False
 *  save\_json： 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）运行test.py传入默认Fasle。运行train.py则传入is\_coco and final\_epoch(一般也是False)
 *  project： 结果保存的项目文件夹路径，默认为“runs/val”
 *  name： 结果保存的文件名，默认为“exp”
 *  exist\_ok： 如果结果保存的文件夹已存在，是否覆盖，默认为False，即不覆盖
 *  half： 是否使用FP16的半精度推理模式，默认为False
 *  dnn： 是否使用OpenCV DNN作为ONNX推理的后端，默认为False
 *  model:  模型， 如果执行val.py就为None 如果执行train.py就会传入ema.ema(ema模型)
 *  dataloader:数据加载器， 如果执行val.py就为None 如果执行train.py就会传入testloader
 *  save\_dir:  文件保存路径， 如果执行val.py就为‘ ’ ，如果执行train.py就会传入save\_dir(runs/train/expn)
 *  plots: 是否可视化，运行val.py传入默认True，运行train.py则传入plots and final\_epoch
 *  callback:  回调函数
 *  compute\_loss: 损失函数，运行val.py传入默认None，运行train.py则传入compute\_loss(train)

### 4.2 初始化/加载模型以及设置设备 

```java
'''======================2.初始化/加载模型以及设置设备====================='''
    # Initialize/load model and set device
    training = model is not None
    if training:  # 通过 train.py 调用的run函数
        # 获得记录在模型中的设备 next为迭代器
        device, pt = next(model.parameters()).device, True

        # 精度减半
        # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

    else:  # 直接通过 val.py 调用 run 函数
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(device, batch_size=batch_size)

        # 路径
        # 调用genera.py中的increment_path函数来生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # mkdir创建路径最后一级目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被32整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 如果不是CPU，使用半进度(图片半精度/模型半精度)
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            # 打印耗时
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        # 调用general.py中的check_dataset函数来检查数据文件是否正常
        data = check_dataset(data)  # check
```

这段代码主要是初始化并加载模型，并设置设备

首先判断模型是否存在，若不存在则训练为假。

接着判断是否是训练时调用run函数——执行train.py， 如果是就使用训练时的设备（一般都是train），如果不是trin.py调用run函数——执行val.py，就调用select\_device选择可用的设备，并生成save\_dir + make dir + 加载模型model + check imgsz + 加载data配置信息。

 *  训练时（train.py）调用：初始化模型参数、训练设备
 *  验证时（val.py）调用：初始化设备、save\_dir文件路径、make dir、加载模型、check imgsz、 加载+check data配置信息

最后判断设备类型并仅仅单GPU支持一半的精度。Half model 只能在单GPU设备上才能使用， 一旦使用half，不但模型需要设为half，输入模型的图片也需要设为half。如果设备类型不是CPU 则将模型由32位浮点数转换为16位浮点数。

### 4.3 加载配置 

```java
'''======================3.加载配置====================='''
    # Configure
    # 将模型转换为测试模式 固定住dropout层和Batch Normalization层
    model.eval()
    # 通过 COCO 数据集的文件夹组织结构判断当前数据集是否为 COCO 数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    # 确定检测的类别数目
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95 的iou向量
    # numel为pytorch预置函数 用来获取张量中的元素个数
    niou = iouv.numel()
```

这段代码主要是加载数据集的yaml配置文件

首先，通过model.eval()  启动模型验证模式，is\_coco判断是否是coco数据集。

然后，确定检测的类别个数nc ，以及计算mAP相关参数，设置iou阈值从0.5-0.95取10个(0.05间隔) 所以iouv: \[0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000\]

### 4.4 加载val数据集 

```java
'''======================4.加载val数据集====================='''
    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            # 创建一张全为0的图片（四维张量）
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       prefix=colorstr(f'{task}: '))[0]
```

这段代码主要是加载val数据集

判断是否是训练。如果不是训练——执行val.py调用run函数，就调用create\_dataloader生成dataloader 。 如果是训练——执行train.py调用run函数，就不需要生成dataloader 可以直接从参数中传过来testloader。

 *  训练时（train.py）调用：加载val数据集
 *  验证时（val.py）调用：不需要加载val数据集 直接从train.py 中传入testloader

### 4.5 初始化 

```java
'''======================5.初始化====================='''
    # 初始化已完成测试的图片数量
    seen = 0
    # 调用matrics中函数 存储混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 调用general.py中的函数  获取coco数据集的类别索引
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化detection中各个指标的值
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化网络训练的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一个类别的AP、图片汇总
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
```

这段代码主要是获取dataloader、初始化模型测试当中用到的相应参数

（1）初始化已完成测试图片数量，设置seen=0

（2）初始化混淆矩阵

（3）获取数据集类名 和coco数据集的类别索引 

（4）设置tqdm进度条的显示信息

（5）初始化p, r, f1, mp, mr, map50, map指标和初始化测试集的损失以及初始化json文件中的字典 统计信息、ap等

### 4.6 验证过程 

#### 4.6.1 开始验证前的预处理 

```java
'''===6.1 开始验证前的预处理==='''
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 获取当前时间
        t1 = time_sync()
        if pt:
            # 将图片数据拷贝到device（GPU）上面
            im = im.to(device, non_blocking=True)
            #对targets也做同样拷贝的操作
            targets = targets.to(device)
        # 将图片从64位精度转换为32位精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # 将图像像素值0-255的范围归一化到0-1的范围
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 四个变量分别代表batchsize、通道数目、图像高度、图像宽度
        nb, _, height, width = im.shape  # batch size, channels, height, width
        # 获取当前时间
        t2 = time_sync()
        # dt[0]: 累计处理数据时间
        dt[0] += t2 - t1
```

这段代码主要是预处理图片和target

获取dataloader当中的batch、图片、目标、路径、形状信息等。

#### 4.6.2 前项推理 

```java
'''===6.2 前向推理==='''
        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        # 累计前向推理时间  dt[1]
        dt[1] += time_sync() - t2
```

这段代码主要是模型前项推理以及累计前项推理时间

 *  out: 推理结果。1个 ，\[bs, anchor\_num\*grid\_w\*grid\_h, xywh+c+20classes\] = \[1, 19200+4800+1200, 25\]
 *  train\_out: 训练结果。3个， \[bs, anchor\_num, grid\_w, grid\_h, xywh+c+20classes\]。如: \[1, 3, 80, 80, 25\] \[1, 3, 40, 40, 25\] \[1, 3, 20, 20, 25\] 

#### 4.6.3 计算损失 

```java
'''===6.3 计算损失==='''
        # Loss
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            # loss 包含bounding box 回归的GIoU、object和class 三者的损失
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
```

这段代码主要是计算验证集损失

判断compute\_loss是否为空，不为空则说明正在执行train.py ，根据传入的compute\_loss计算损失值。

loss 包含bounding box 回归的GIoU、object和class 三者的损失

 *  分类损失(cls\_loss)：该损失用于判断模型是否能够准确地识别出图像中的对象，并将其分类到正确的类别中。
 *  置信度损失(obj\_loss)：该损失用于衡量模型预测的框（即包含对象的矩形）与真实框之间的差异。
 *  边界框损失(box\_loss)：该损失用于衡量模型预测的边界框与真实边界框之间的差异，这有助于确保模型能够准确地定位对象。

#### 4.6.4 NMS获得预测框 

```java
'''===6.4 NMS获得预测框==='''
        # NMS
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # 提取bach中每一张图片的目标的label
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        # 计算NMS过程所需要的时间
        t3 = time_sync()
        # 调用general.py中的函数 进行非极大值抑制操作
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 累计NMS时间
        dt[2] += time_sync() - t3
```

这段代码主要是运行NMS 目标检测的后处理模块，用于删除冗余的bounding box  
首先将真实框target的xywh (因为 target 是在 labelimg 中做了归一化的)映射到真实的图像尺寸  
然后，在 NMS之前将数据集标签 targets 添加到模型预测中，这允许在数据集中自动标记(for autolabelling)其它对象(在pred中混入gt)并且mAP反映了新的混合标签。

最后调用general.py中的函数，进行NMS操作，并计算NMS过程所需要的时间，

#### 4.6.5 统计真实框、预测框信息 

```java
'''===6.5 统计真实框、预测框信息==='''
        # Metrics
        # si代表第si张图片，pred是对应图片预测的label信息
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            # nl为图片检测到的目标个数
            nl = len(labels)
            # tcls为检测到的目标的类别 label矩阵的第一列
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # 第si张图片对应的文件路径
            path, shape = Path(paths[si]), shapes[si][0]
            # 统计测试图片数量 +1
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl: # 预测为空但同时有label信息
                    # stats初始化为一个空列表[] 此处添加一个空信息
                    # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            # 预测
            if single_cls:
                pred[:, 5] = 0
            # 对pred进行深复制
            predn = pred.clone()
            # 调用general.py中的函数 将图片调整为原图大小
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            # 预测框评估
            if nl:
                # 获得xyxy格式的框
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 调用general.py中的函数 将图片调整为原图大小
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 处理完gt的尺寸信息，重新构建成 (cls, xyxy)的格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space label
                # 对当前的预测框与gt进行一一匹配，并且在预测框的对应位置上获取iou的评分信息，其余没有匹配上的预测框设置为False
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 计算混淆矩阵 confusion_matrix
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                # 返回一个形状为为pred.shape[0, 类型为torch.dtype，里面的每一个值都是0的tensor
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            # 保存预测信息到txt文件
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # 保存预测信息到json字典
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
```

这段代码主要是统计每张图片真实框和预测框的相关信息

首先统计每张图片的相关信息，如预测label信息、标签gt信息等。然后统计检测到的目标个数和类别以及相对应的文件路径。

接着利用得到的上述信息进行目标的预测，并将结果保存同时输出日志，分别保存预测信息到image\_name.txt文件和coco格式的json字典。

 *  txt文件保存的预测信息：cls＋xywh＋conf
 *  jdict字典保存的预测信息：image\_id + category\_id + bbox + score

#### 4.6.6 画出前三个batch图片的gt和pred框 

```java
'''===6.6 画出前三个batch图片的gt和pred框==='''
        # Plot images
        # 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            '''
              Thread()函数为创建一个新的线程来执行这个函数 函数为plots.py中的plot_images函数
              target: 执行的函数  args: 传入的函数参数  daemon: 当主线程结束后, 由他创建的子线程Thread也已经自动结束了
              .start(): 启动线程  当thread一启动的时候, 就会运行我们自己定义的这个函数plot_images
              如果在plot_images里面打开断点调试, 可以发现子线程暂停, 但是主线程还是在正常的训练(还是正常的跑)
            '''
            # 传入plot_images函数之前需要改变pred的格式  target则不需要改
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()
```

这段代码主要是创建子进程进行绘图，画出前三个batch图片的gt和pred框

 *  gt : 真实框，Ground truth box, 是人工标注的位置，存放在标注文件中
 *  pred : 预测框，Prediction box， 是由目标检测模型计算输出的框

#### 4.6.7 计算指标 

```java
'''===6.7 计算指标==='''
    # Compute metrics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为对应格式numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 计算上述测试过程中的各种性能指标
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
      
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
       
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
       
    else:
        nt = torch.zeros(1)
```

这段代码主要是计算评判分类效果的各种指标

correct \[img\_sum, 10\] ：整个数据集所有图片中所有预测框在每一个iou条件下是否是TP \[1905, 10\]

 *  p: \[nc\] 最大平均f1时每个类别的precision
 *  r:  \[nc\] 最大平均f1时每个类别的recall
 *  ap: \[71, 10\] 数据集每个类别在10个iou阈值下的mAP
 *  f1： \[nc\] 最大平均f1时每个类别的f1
 *  ap\_class: \[nc\] 返回数据集中所有的类别index

conf \[img\_sum\] ：整个数据集所有图片中所有预测框的conf \[1905\]

 *  ap50:  \[nc\] 所有类别的mAP@0.5 
 *  ap:\[nc\] 所有类别的mAP@0.5:0.95

pcls \[img\_sum\] ：整个数据集所有图片中所有预测框的类别 \[1905\]

 *  mp:  \[1\] 所有类别的平均precision(最大f1时)
 *  mr: \[1\] 所有类别的平均recall(最大f1时)
 *  map50: \[1\] 所有类别的平均mAP@0.5
 *  map: \[1\] 所有类别的平均mAP@0.5:0.95

tcls \[gt\_sum\] ：整个数据集所有图片所有gt框的class \[929\]

 *  nt:  \[nc\] 统计出整个数据集的gt框中数据集各个类别的个数

#### 4.6.8 打印日志 

```java
'''===6.8 打印日志==='''
    # Print results
    # 按照以下格式来打印测试过程的指标
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 打印每一个类别对应的性能指标
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印 推断/NMS过程/总过程 的在每一个batch上面的时间消耗
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
```

这段代码主要是打印各项指标

首先打印测试过程中的指标，包括：数据集图片数量 、 数据集gt框的数量 、所有类别的平均precision 、 所有类别的平均recall 、所有类别的平均mAP@0.5 、所有类别的平均mAP@0.5:0.95  
然后打印每个类别对应的指标，包括：类别、数据集图片数量 、这个类别的gt框数量、这个类别的precision、这个类别的recall、这个类别的mAP@0.5、这个类别的mAP@0.5:0.95  
最后打印前向传播耗费的总时间、nms耗费总时间、总时间

#### 4.6.9 保存验证结果 

```java
'''===6.9 保存验证结果==='''
    # Plots
    # 绘图
    if plots:
        # confusion_matrix.plot（）函数绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 调用Loggers中的on_val_end方法，将日志记录并生成一些记录的图片
        callbacks.run('on_val_end')

    # Save JSON
    # 采用之前保存的json文件格式预测结果 通过coco的api评估各个指标
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # 注释的json格式
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 预测的json格式
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        # 在控制台打印coco的api评估各个指标，保存到json文件
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        # 打开pred_json文件只用于写入
        with open(pred_json, 'w') as f: # w:打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
            # 测试集的标签也需要转成coco的json格式。将 dict==>json 序列化，用json.dumps()
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            # 以下过程为利用官方coco工具进行结果的评测
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
```

这段代码主要是绘制混淆矩阵和利用cocoapi进行相关性能指标的评估

首先用confusion\_matrix.plot（）函数绘制混淆矩阵

confusion\_matrix.png:

![](https://i-blog.csdnimg.cn/blog_migrate/610d6440a790266aa6be118c02204a18.png)

接着采用之前保存的json文件格式预测结果，通过cocoapi评估各个指标，需要注意的是测试集的标签也要转为coco的json格式。另外，因为coco测试集的标签是给出的，因此这个评估过程结合了测试集标签 ，不过在更多的目标检测场合下，为保证公正测试集标签不会给出。

#### 4.6.10 返回结果 

```java
'''===6.10 返回结果==='''
    # Return results
    # 返回测试指标结果
    model.float() # 将模型转换为适用于训练的状态
    if not training:# 如果不是训练过程则将结果保存到对应的路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 在控制台中打印保存结果
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # 返回对应的测试结果
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
```

这段代码主要是返回对应的测试结果

 *  mp： \[1\] 所有类别的平均precision(最大f1时)
 *  mr： \[1\] 所有类别的平均recall(最大f1时)
 *  map50：  \[1\] 所有类别的平均mAP@0.5
 *  map ： \[1\] 所有类别的平均mAP@0.5:0.95
 *  val\_box\_loss ： \[1\] 验证集回归损失
 *  val\_obj\_loss： \[1\] 验证集置信度损失
 *  val\_cls\_loss：  \[1\] 验证集分类损失 maps: \[80\] 所有类别的mAP@0.5:0.95 t: \{tuple: 3\}
    
     *  0: 打印前向传播耗费的总时间
     *  1:  nms耗费总时间
     *  2:  总时间

## 🚀五、设置opt参数 

```java
'''===============================================五、设置opt参数==================================================='''
def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # 模型的权重文件地址yolov5s.pt
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    # 前向传播的批次大小 默认32
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    # 输入网络的图片分辨率 默认640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # object置信度阈值 默认0.001
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # 测试的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 数据集是否只用一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # 测试是否使用TTA Test Time Augment 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存label+prediction杂交结果到对应.txt，默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    # 测试保存的源文件 默认runs/val
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # 测试保存的文件地址 默认exp  保存在runs/val/exp下
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 是否使用半精度推理 默认False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN对ONNX 模型推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # 解析上述参数
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
```

opt参数解析： 

 *  data： 数据集文件的路径，默认为COCO128数据集的配置文件路径
 *  weights： 模型权重文件的路径，默认为YOLOv5s的权重文件路径
 *  batch\_size: 前向传播的批次大小，运行val.py传入默认32 。运行train.py则传入batch\_size // WORLD\_SIZE \* 2
 *  imgsz： 输入图像的大小，默认为640x640
 *  conf\_thres： 置信度阈值，默认为0.001
 *  iou\_thres： 非极大值抑制的iou阈值，默认为0.6
 *  task:  设置测试的类型 有train, val, test, speed or study几种，默认val
 *  device： 使用的设备类型，默认为空，表示自动选择最合适的设备
 *  single\_cls:  数据集是否只用一个类别，运行val.py传入默认False 运行train.py则传入single\_cls
 *  augment： 是否使用数据增强的方式进行检测，默认为False
 *  verbose:  是否打印出每个类别的mAP，运行val.py传入默认Fasle。运行train.py则传入nc < 50 and final\_epoch
 *  save\_txt： 是否将检测结果保存为文本文件，默认为False
 *  save\_hybrid: 是否保存 label+prediction hybrid results to \*.txt 默认False
 *  save\_conf： 是否在保存的文本文件中包含置信度信息，默认为False
 *  save\_json： 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）运行test.py传入默认Fasle。运行train.py则传入is\_coco and final\_epoch(一般也是False)
 *  project： 结果保存的项目文件夹路径，默认为“runs/val”
 *  name： 结果保存的文件名，默认为“exp”
 *  exist\_ok： 如果结果保存的文件夹已存在，是否覆盖，默认为False，即不覆盖
 *  half： 是否使用FP16的半精度推理模式，默认为False
 *  dnn： 是否使用OpenCV DNN作为ONNX推理的后端，默认为False

（关于调参，推荐大家看@迪菲赫尔曼大佬的这篇文章：[手把手带你调参YOLOv5 (v5.0-v7.0)（验证）\_迪菲赫尔曼的博客-CSDN博客][YOLOv5 _v5.0-v7.0_-CSDN]）

## 🚀六、执行main()函数 

```java
'''==============================六、执行main（）函数======================================'''
def main(opt):
    # 检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 如果opt.task == 'speed' 就测试yolov5系列和yolov3-spp各个模型的速度评估
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        # 如果opt.task = ['study']就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # 保存的文件名
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                # x坐标轴和y坐标
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    # 返回相关结果和时间
                    y.append(r + t)  # results and times
                # 将y输出保存
                np.savetxt(f, y, fmt='%10.4g')  # save
            # 命令行执行命令将study文件进行压缩
            os.system('zip -r study.zip study_*.txt')
            # 调用plots.py中的函数 可视化各个指标
            plot_val_study(x=x)  # plot

# python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
```

这段代码主要是根据解析的opt参数，调用run函数

这个模块根据opt.task可以分为三个分支，主要的分支还是在 opt.task in (‘train’, ‘val’, ‘test’)。而其他的两个分支，大家大概看看在干什么就可以了，没什么用。一般我们都是直接进入第一个分支，执行run（）函数。

## 🚀七、val.py代码全部注释 

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""
'''===============================================一、导入包==================================================='''
'''======================1.导入安装好的python库====================='''

import argparse # 解析命令行参数的库
import json # 实现字典列表和JSON字符串之间的相互解析
import os  # 与操作系统进行交互的文件库 包含文件路径操作与解析
import sys # sys系统模块 包含了与Python解释器和它的环境有关的函数
from pathlib import Path # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread # python中处理多线程的库

import numpy as np # 矩阵计算基础库
import torch # pytorch 深度学习库
from tqdm import tqdm  # 用于直观显示进度条的一个库

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()# __file__指的是当前文件(即val.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/val.py
ROOT = FILE.parents[0]  # YOLOv5 root directory ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path: # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH 把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径

'''===================3..加载自定义模块============================'''
from models.common import DetectMultiBackend # yolov5的网络结构(yolov5)
from utils.callbacks import Callbacks # 和日志相关的回调函数
from utils.datasets import create_dataloader # 加载数据集的函数
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)  # 定义了一些常用的工具函数
from utils.metrics import ConfusionMatrix, ap_per_class # 在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权
from utils.plots import output_to_target, plot_images, plot_val_study # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.torch_utils import select_device, time_sync  # 定义了一些与PyTorch有关的工具函数

'''===============================================二、保存信息==================================================='''
'''======================1.保存预测信息到txt文件====================='''
def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # gn = [w, h, w, h] 对应图片的宽高  用于后面归一化
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
    for *xyxy, conf, cls in predn.tolist():
        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 保存预测类别和坐标值到对应图片image_name.txt文件中
        with open(file, 'a') as f:
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n')

'''======================2.保存预测信息到coco格式的json字典====================='''
def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    # 获取图片id
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 获取预测框 并将xyxy转为xywh格式
    box = xyxy2xywh(predn[:, :4])  # xywh
    # 之前的的xyxy格式是左上角右下角坐标  xywh是中心的坐标和宽高
    # 而coco的json格式的框坐标是xywh(左上角坐标 + 宽高)
    # 所以这行代码是将中心点坐标 -> 左上角坐
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    # 序列解包
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})
    '''
     image_id: 图片id 即属于哪张图片
     category_id: 类别 coco91class()从索引0~79映射到索引0~90
     bbox: 预测框坐标
     score: 预测得分
    '''
'''===============================================三、计算指标==================================================='''
def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    返回每个预测框在10个IoU阈值上是TP还是FP
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # 构建一个[pred_nums, 10]全为False的矩阵
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # 计算每个gt与每个pred的iou，shape为: [gt_nums, pred_nums]
    iou = box_iou(labels[:, 1:], detections[:, :4])
    '''
    首先iou >= iouv[0]：挑选出iou>0.5的所有预测框，进行筛选,shape为: [gt_nums, pred_nums]
    同时labels[:, 0:1] == detections[:, 5]：构建出一个预测类别与真实标签是否相同的矩阵表, shape为: [gt_nums, pred_nums]
    只有同时符合以上两点条件才被赋值为True，此时返回当前矩阵的一个行列索引，x是两个元祖x1,x2
    点(x[0][i], x[1][i])就是符合条件的预测框
    '''
    # iou超过阈值而且类别正确，则为True，返回索引
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    # 如果存在符合条件的预测框
    if x[0].shape[0]: # 至少有一个TP
        # 将符合条件的位置构建成一个新的矩阵，第一列是行索引（表示gt索引），第二列是列索引（表示预测框索引），第三列是iou值
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            # argsort获得有小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引，对matches矩阵进行排序
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            '''
            参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
            matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
            '''
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            '''
            matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
            '''
            # 以上操作实现了为每一个gt分配一个iou最高的类别的预测框，实现一一对应

        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        '''
         当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
         需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iouv的评价指标
        '''
        # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
    return correct

'''===============================================四、run()函数==================================================='''
'''======================1.设置参数====================='''
@torch.no_grad()
def run(data, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
        weights=None,  # 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s
        batch_size=32,  # 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
        imgsz=640,  # 输入网络的图片分辨率 运行test.py传入默认640 运行train.py则传入imgsz_test
        conf_thres=0.001,  # object置信度阈值 默认0.001
        iou_thres=0.6,  # 进行NMS时IOU的阈值 默认0.6
        task='val',  # 设置测试的类型 有train, val, test, speed or study几种 默认val
        device='',  # 执行 val.py 所在的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # dataloader中的最大 worker 数（线程个数）
        augment=False,  # 数据集是否只有一个类别 默认False
        verbose=False,  # 是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
        save_txt=False,  # 是否以txt文件的形式保存模型预测框的坐标 默认True
        save_hybrid=False,  # 是否保存预测每个目标的置信度到预测txt文件中 默认True
        save_conf=False,  # 保存置信度
        save_json=False,  # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）,
                      #运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
        project=ROOT / 'runs/val',  # 验证结果保存的根目录 默认是 runs/val
        name='exp',  # 验证结果保存的目录 默认是exp  最终: runs/val/exp
        exist_ok=False,  # 如果文件存在就increment name，不存在就新建  默认False(默认文件都是不存在的)
        half=True,  # 使用 FP16 的半精度推理
        dnn=False,  # 在 ONNX 推理时使用 OpenCV DNN 后段端
        model=None,  # 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
        dataloader=None, # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
        save_dir=Path(''), # 文件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn)
        plots=True, # 是否可视化 运行val.py传入，默认True
        callbacks=Callbacks(),
        compute_loss=None, # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
        ):
    '''======================2.初始化/加载模型以及设置设备====================='''
    # Initialize/load model and set device
    training = model is not None
    if training:  # 通过 train.py 调用的run函数
        # 获得记录在模型中的设备 next为迭代器
        device, pt = next(model.parameters()).device, True

        # 精度减半
        # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

        # 如果不是train.py调用run函数(执行val.py脚本)就调用select_device选择可用的设备
        # 并生成save_dir + make dir + 加载模型model + check imgsz + 加载data配置信息
    else:  # 直接通过 val.py 调用 run 函数
        # 调用torch_utils中select_device来选择执行程序时的设备
        device = select_device(device, batch_size=batch_size)

        # 路径
        # 调用genera.py中的increment_path函数来生成save_dir文件路径  run\test\expn
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # mkdir创建路径最后一级目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # 加载模型  只在运行test.py才需要自己加载model
        # 加载模型为32位浮点数模型（权重参数） 调用experimental.py文件中的attempt_load函数
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被32整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # 如果不是CPU，使用半进度(图片半精度/模型半精度)
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            # 打印耗时
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        # 调用general.py中的check_dataset函数来检查数据文件是否正常
        data = check_dataset(data)  # check

    '''======================3.加载配置====================='''
    # Configure
    # 将模型转换为测试模式 固定住dropout层和Batch Normalization层
    model.eval()
    # 通过 COCO 数据集的文件夹组织结构判断当前数据集是否为 COCO 数据集
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    # 确定检测的类别数目
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 计算mAP相关参数
    # 设置iou阈值 从0.5-0.95取10个(0.05间隔)   iou vector for mAP@0.5:0.95
    # iouv: [0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000]
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95 的iou向量
    # numel为pytorch预置函数 用来获取张量中的元素个数
    niou = iouv.numel()

    '''======================4.加载val数据集====================='''
    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            # 创建一张全为0的图片（四维张量）
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       prefix=colorstr(f'{task}: '))[0]

    '''======================5.初始化====================='''
    # 初始化已完成测试的图片数量
    seen = 0
    # 调用matrics中函数 存储混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取数据集所有类别的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 调用general.py中的函数  获取coco数据集的类别索引
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化detection中各个指标的值
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化网络训练的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一个类别的AP、图片汇总
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar

    '''======================6.开始验证====================='''
    '''===6.1 开始验证前的预处理==='''
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 获取当前时间
        t1 = time_sync()
        if pt:
            # 将图片数据拷贝到device（GPU）上面
            im = im.to(device, non_blocking=True)
            #对targets也做同样拷贝的操作
            targets = targets.to(device)
        # 将图片从64位精度转换为32位精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # 将图像像素值0-255的范围归一化到0-1的范围
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 四个变量分别代表batchsize、通道数目、图像高度、图像宽度
        nb, _, height, width = im.shape  # batch size, channels, height, width
        # 获取当前时间
        t2 = time_sync()
        # dt[0]: 累计处理数据时间
        dt[0] += t2 - t1

        '''===6.2 前向推理==='''
        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        # 累计前向推理时间  dt[1]
        dt[1] += time_sync() - t2

        '''===6.3 计算损失==='''
        # Loss
        # compute_loss不为空 说明正在执行train.py  根据传入的compute_loss计算损失值
        if compute_loss:
            # loss 包含bounding box 回归的GIoU、object和class 三者的损失
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        '''===6.4 NMS获得预测框==='''
        # NMS
        # 运行NMS 目标检测的后处理模块 用于删除冗余的bounding box
        # targets: [num_target, img_index+class_index+xywh] = [31, 6]
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # 提取bach中每一张图片的目标的label
        # lb: {list: bs} 第一张图片的target[17, 5] 第二张[1, 5] 第三张[7, 5] 第四张[6, 5]
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        # 计算NMS过程所需要的时间
        t3 = time_sync()
        # 调用general.py中的函数 进行非极大值抑制操作
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 累计NMS时间
        dt[2] += time_sync() - t3

        '''===6.5 统计真实框、预测框信息==='''
        # Metrics
        # 为每张图片做统计，将写预测信息到txt文件，生成json文件字典，统计tp等
        # out: list{bs}  [300, 6] [42, 6] [300, 6] [300, 6]  [:, image_index+class+xywh]
        # si代表第si张图片，pred是对应图片预测的label信息
        for si, pred in enumerate(out):
            # 获取第si张图片的gt标签信息 包括class, x, y, w, h    target[:, 0]为标签属于哪张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            # nl为图片检测到的目标个数
            nl = len(labels)
            # tcls为检测到的目标的类别 label矩阵的第一列
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # 第si张图片对应的文件路径
            path, shape = Path(paths[si]), shapes[si][0]
            # 统计测试图片数量 +1
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl: # 预测为空但同时有label信息
                    # stats初始化为一个空列表[] 此处添加一个空信息
                    # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            # 预测
            if single_cls:
                pred[:, 5] = 0
            # 对pred进行深复制
            predn = pred.clone()
            # 调用general.py中的函数 将图片调整为原图大小
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            # 预测框评估
            if nl:
                # 获得xyxy格式的框
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 调用general.py中的函数 将图片调整为原图大小
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # 处理完gt的尺寸信息，重新构建成 (cls, xyxy)的格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space label
                # 对当前的预测框与gt进行一一匹配，并且在预测框的对应位置上获取iou的评分信息，其余没有匹配上的预测框设置为False
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 计算混淆矩阵 confusion_matrix
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                # 返回一个形状为为pred.shape[0, 类型为torch.dtype，里面的每一个值都是0的tensor
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            # 保存预测信息到txt文件
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            # 保存预测信息到json字典
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        '''===6.6 画出前三个batch图片的 gt 和 pred 框==='''
        # Plot images
        # 画出前三个batch的图片的ground truth和预测框predictions(两个图)一起保存
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            '''
              Thread()函数为创建一个新的线程来执行这个函数 函数为plots.py中的plot_images函数
              target: 执行的函数  args: 传入的函数参数  daemon: 当主线程结束后, 由他创建的子线程Thread也已经自动结束了
              .start(): 启动线程  当thread一启动的时候, 就会运行我们自己定义的这个函数plot_images
              如果在plot_images里面打开断点调试, 可以发现子线程暂停, 但是主线程还是在正常的训练(还是正常的跑)
            '''
            # 传入plot_images函数之前需要改变pred的格式  target则不需要改
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    '''===6.7 计算指标==='''
    # Compute metrics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为对应格式numpy
    # stats[0].any(): stats[0]是否全部为False, 是则返回 False, 如果有一个为 True, 则返回 True
    if len(stats) and stats[0].any():
        # 计算上述测试过程中的各种性能指标
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        '''
        根据上面的统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）等指标
        p: [nc] 最大平均f1时每个类别的precision
        r: [nc] 最大平均f1时每个类别的recall
        ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        f1 [nc] 最大平均f1时每个类别的f1
        ap_class: [nc] 返回数据集中所有的类别index
        '''
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        '''
        ap50: [nc] 所有类别的mAP@0.5   
        ap: [nc] 所有类别的mAP@0.5:0.95 
        '''
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        '''
         mp: [1] 所有类别的平均precision(最大f1时)
         mr: [1] 所有类别的平均recall(最大f1时)
         map50: [1] 所有类别的平均mAP@0.5
         map: [1] 所有类别的平均mAP@0.5:0.95
        '''
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        '''
         nt: [nc] 统计出整个数据集的gt框中数据集各个类别的个数
        '''
    else:
        nt = torch.zeros(1)

    '''===6.8 打印日志==='''
    # Print results
    # 按照以下格式来打印测试过程的指标
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 打印每一个类别对应的性能指标
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印 推断/NMS过程/总过程 的在每一个batch上面的时间消耗
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    '''===6.9 保存验证结果==='''
    # Plots
    # 绘图
    if plots:
        # confusion_matrix.plot（）函数绘制混淆矩阵
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 调用Loggers中的on_val_end方法，将日志记录并生成一些记录的图片
        callbacks.run('on_val_end')

    # Save JSON
    # 采用之前保存的json文件格式预测结果 通过coco的api评估各个指标
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # 注释的json格式
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        # 预测的json格式
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        # 在控制台打印coco的api评估各个指标，保存到json文件
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        # 打开pred_json文件只用于写入
        with open(pred_json, 'w') as f: # w:打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
            # 测试集的标签也需要转成coco的json格式。将 dict==>json 序列化，用json.dumps()
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            # 以下过程为利用官方coco工具进行结果的评测
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    '''===6.10 返回结果==='''
    # Return results
    # 返回测试指标结果
    model.float() # 将模型转换为适用于训练的状态
    if not training:# 如果不是训练过程则将结果保存到对应的路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 在控制台中打印保存结果
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # 返回对应的测试结果
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
'''
 (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()): {tuple:7}
          0: mp [1] 所有类别的平均precision(最大f1时)
          1: mr [1] 所有类别的平均recall(最大f1时)
          2: map50 [1] 所有类别的平均mAP@0.5
          3: map [1] 所有类别的平均mAP@0.5:0.95
          4: val_box_loss [1] 验证集回归损失
          5: val_obj_loss [1] 验证集置信度损失
          6: val_cls_loss [1] 验证集分类损失
     maps: [80] 所有类别的mAP@0.5:0.95
     t: {tuple: 3} 0: 打印前向传播耗费的总时间   1: nms耗费总时间   2: 总时间
'''

'''===============================================五、设置opt参数==================================================='''
def parse_opt():
    parser = argparse.ArgumentParser()
    # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # 模型的权重文件地址yolov5s.pt
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    # 前向传播的批次大小 默认32
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    # 输入网络的图片分辨率 默认640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # object置信度阈值 默认0.001
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # 测试的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 数据集是否只用一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # 测试是否使用TTA Test Time Augment 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 保存label+prediction杂交结果到对应.txt，默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    # 测试保存的源文件 默认runs/val
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # 测试保存的文件地址 默认exp  保存在runs/val/exp下
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 是否使用半精度推理 默认False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN对ONNX 模型推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # 解析上述参数
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt

'''===============================================六、执行main（）函数==================================================='''
def main(opt):
    # 检测requirements文件中需要的包是否安装好了
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # 如果task in ['train', 'val', 'test']就正常测试 训练集/验证集/测试集
    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        # 如果opt.task == 'speed' 就测试yolov5系列和yolov3-spp各个模型的速度评估
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        # 如果opt.task = ['study']就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # 保存的文件名
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                # x坐标轴和y坐标
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    # 返回相关结果和时间
                    y.append(r + t)  # results and times
                # 将y输出保存
                np.savetxt(f, y, fmt='%10.4g')  # save
            # 命令行执行命令将study文件进行压缩
            os.system('zip -r study.zip study_*.txt')
            # 调用plots.py中的函数 可视化各个指标
            plot_val_study(x=x)  # plot

# python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
```

本文参考：

[【YOLOV5-5.x 源码解读】满船清梦压星河HK的博客-CSDN博客][YOLOV5-5.x _HK_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/fcd892a91263916ea458b8589e579a5d.gif)


[mirrors _ ultralytics _ yolov5 _ GitCode]: https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator
[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764
[Link 1]: #%E5%89%8D%E8%A8%80%C2%A0
[Link 2]: #main-toc
[Link 3]: #%E2%80%8B%E7%BC%96%E8%BE%91%C2%A0%F0%9F%9A%80%E4%B8%80%E3%80%81%E5%AF%BC%E5%8C%85%E4%B8%8E%E5%9F%BA%E6%9C%AC%E9%85%8D%E7%BD%AE
[1.1 _python]: #1.1%20%E5%AF%BC%E5%85%A5%E5%AE%89%E8%A3%85%E5%A5%BD%E7%9A%84python%E5%BA%93
[1.2]: #1.2%20%E8%8E%B7%E5%8F%96%E5%BD%93%E5%89%8D%E6%96%87%E4%BB%B6%E7%9A%84%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84
[1.3]: #1.4%20%E5%8A%A0%E8%BD%BD%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97
[Link 4]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E4%BF%9D%E5%AD%98%E4%BF%A1%E6%81%AF
[2.1 _txt]: #2.1%20%E4%BF%9D%E5%AD%98%E9%A2%84%E6%B5%8B%E4%BF%A1%E6%81%AF%E5%88%B0txt%E6%96%87%E4%BB%B6
[2.2 _coco_json]: #2.2%20%E4%BF%9D%E5%AD%98%E9%A2%84%E6%B5%8B%E4%BF%A1%E6%81%AF%E5%88%B0coco%E6%A0%BC%E5%BC%8F%E7%9A%84json%E5%AD%97%E5%85%B8
[Link 5]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E8%AE%A1%E7%AE%97%E6%8C%87%E6%A0%87%C2%A0
[run]: #%F0%9F%9A%80%E5%9B%9B%E3%80%81%E6%89%A7%E8%A1%8Crun%EF%BC%88%EF%BC%89%E5%87%BD%E6%95%B0
[4.1]: #4.1%20%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0
[4.2]: #4.2%C2%A0%E5%88%9D%E5%A7%8B%E5%8C%96%2F%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B%E4%BB%A5%E5%8F%8A%E8%AE%BE%E7%BD%AE%E8%AE%BE%E5%A4%87
[4.3]: #4.3%20%E5%8A%A0%E8%BD%BD%E9%85%8D%E7%BD%AE
[4.4 _val]: #4.4%C2%A0%E5%8A%A0%E8%BD%BDval%E6%95%B0%E6%8D%AE%E9%9B%86
[4.5]: #%C2%A04.5%C2%A0%E5%88%9D%E5%A7%8B%E5%8C%96
[4.6]: #4.6%20%E9%AA%8C%E8%AF%81%E8%BF%87%E7%A8%8B
[4.6.1]: #4.6.1%C2%A0%E5%BC%80%E5%A7%8B%E9%AA%8C%E8%AF%81%E5%89%8D%E7%9A%84%E9%A2%84%E5%A4%84%E7%90%86
[4.6.2]: #4.6.2%20%E5%89%8D%E9%A1%B9%E6%8E%A8%E7%90%86
[4.6.3]: #4.6.3%20%E8%AE%A1%E7%AE%97%E6%8D%9F%E5%A4%B1
[4.6.4 NMS]: #%C2%A04.6.4%20NMS%E8%8E%B7%E5%BE%97%E9%A2%84%E6%B5%8B%E6%A1%86
[4.6.5]: #4.6.5%20%E7%BB%9F%E8%AE%A1%E7%9C%9F%E5%AE%9E%E6%A1%86%E3%80%81%E9%A2%84%E6%B5%8B%E6%A1%86%E4%BF%A1%E6%81%AF
[4.6.6 _batch_gt_pred]: #4.6.6%20%E7%94%BB%E5%87%BA%E5%89%8D%E4%B8%89%E4%B8%AAbatch%E5%9B%BE%E7%89%87%E7%9A%84gt%E5%92%8Cpred%E6%A1%86
[4.6.7]: #4.6.7%20%E8%AE%A1%E7%AE%97%E6%8C%87%E6%A0%87
[4.6.8 _]: #4.6.8%20%E6%89%93%E5%8D%B0%E6%97%A5%E5%BF%97%C2%A0%20%C2%A0
[4.6.9 _]: #4.6.9%20%E4%BF%9D%E5%AD%98%E9%AA%8C%E8%AF%81%E7%BB%93%E6%9E%9C%C2%A0%20%C2%A0
[4.6.10]: #4.6.10%20%E8%BF%94%E5%9B%9E%E7%BB%93%E6%9E%9C
[opt]: #%F0%9F%9A%80%E4%BA%94%E3%80%81%E8%AE%BE%E7%BD%AEopt%E5%8F%82%E6%95%B0
[main]: #%F0%9F%9A%80%E5%85%AD%E3%80%81%E6%89%A7%E8%A1%8Cmain%28%29%E5%87%BD%E6%95%B0
[val.py]: #%F0%9F%9A%80%E5%85%AD%E3%80%81train.py%E4%BB%A3%E7%A0%81%E5%85%A8%E9%83%A8%E6%B3%A8%E9%87%8A
[YOLOv5 _v5.0-v7.0_-CSDN]: https://yolov5.blog.csdn.net/article/details/126630836
[YOLOV5-5.x _HK_-CSDN]: https://blog.csdn.net/qq_38253797/article/details/119577291?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167892894516800182182993%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167892894516800182182993&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-119577291-null-null.142%5Ev73%5Ewechat,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=YOLOv5val&spm=1018.2226.3001.4187