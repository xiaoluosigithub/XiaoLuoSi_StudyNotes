#### ![](https://i-blog.csdnimg.cn/blog_migrate/7ec31499fc292ba5d52674a97a23df32.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/6e26b415723e09a04746856844516326.jpeg)

## 前言 

本篇文章主要是对YOLOv5项目的训练部分train.py。通常这个文件主要是用来读取用户自己的数据集，加载模型并训练。

文章代码逐行手打注释，每个模块都有对应讲解，一文帮你梳理整个代码逻辑！

友情提示：全文近5万字，可以先点![](https://i-blog.csdnimg.cn/blog_migrate/6d715f6e5e1fc12bc794cecabeab9d20.gif)再慢慢看哦~

源码下载地址：[mirrors / ultralytics / yolov5 · GitCode][mirrors _ ultralytics _ yolov5 _ GitCode]

![](https://i-blog.csdnimg.cn/blog_migrate/8879b693f41f11bb0e599f89255ee995.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/d96b4c26ddc180982671cf587cefbd65.gif)🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]  
[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]  
[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

## 目录 

[前言][Link 1]

[目录][Link 2]

[🚀一、导包和基本配置][Link 3]

[1.1 Usage][]

[1.2 导入安装好的python库][1.2 _python]

[1.3 获取当前文件的绝对路径][1.3]

[1.4 加载自定义模块][1.4]

[1.5 分布式训练初始化][1.5]

[🚀二、执行main（）函数][main]

[2.1 检查分布式训练环境][2.1]

[2.2 判断是否断点续训][2.2]

[2.3 判断是否分布式训练][2.3]

[2.4 判断是否进化训练][2.4]

[🚀三、设置opt参数][opt]

[🚀四、执行train（）函数][train]

[4.1 加载参数和初始化配置信息][4.1]

[4.1.1 载入参数][4.1.1]

[4.1.2 创建训练权重目录和保存路径][4.1.2]

[4.1.3 读取超参数配置文件][4.1.3]

[4.1.4 设置参数的保存路径][4.1.4]

[4.1.5 加载日志信息][4.1.5]

[4.1.6 加载其它参数][4.1.6]

[4.2 加载网络模型][4.2]

[4.2.1 加载预训练模型][4.2.1]

[4.2.2 设置冻结层][4.2.2]

[ 4.2.3 设置优化器][4.2.3]

[4.2.4 设置学习率][4.2.4]

[4.2.5 训练前最后准备][4.2.5]

[4.3 加载数据集][4.3]

[4.3.1 创建数据集][4.3.1]

[4.3.2 计算anchor][4.3.2 _anchor]

[4.4 训练过程][4.4]

[4.4.1 初始化训练需要的模型参数][4.4.1]

[4.4.2 训练热身部分][4.4.2]

[4.4.3 开始训练][4.4.3]

[4.4.4 训练完成保存模型][4.4.4]

[4.5 打印信息并释放显存][4.5]

[🚀五、执行run（）函数][run]

[🚀六、train.py代码全部注释][train.py]

![](https://i-blog.csdnimg.cn/blog_migrate/9653e58f9aa0e8d94952bc28c8875d36.gif)

## 🚀一、导包和基本配置 

### 1.1 Usage 

```java
"""
Train a YOLOv5 model on a custom dataset
在数据集上训练 yolo v5 模型
Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    训练数据为coco128 coco128数据集中有128张图片 80个类别，是规模较小的数据集
"""
```

这里是开头作者注释的一个部分，意在说明一些项目基本情况。  
第一行表示我们用的模型是YOLOv5；

第二行表示我们传入的data数据集是coco128数据集，有128张图片，80个类别，使用的权重模型是yolov5s模型，–img表示图片大小640。

### 1.2 导入安装好的python库 

```java
'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import math  # 数学公式模块
import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time   # 时间模块 更底层
from copy import deepcopy  # 深度拷贝模块
from datetime import datetime  # datetime模块能以更方便的格式显示日期或对日期进行运算。
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np  # numpy数组操作模块
import torch # 引入torch
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import yaml  # yaml是一种直观的能够被电脑识别的的数据序列化格式，容易被人类阅读，并且容易和脚本语言交互。一般用于存储配置文件。
from torch.cuda import amp  # PyTorch amp自动混合精度训练模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 多卡训练模块
from torch.optim import SGD, Adam, lr_scheduler   # tensorboard模块
from tqdm import tqdm  # 进度条模块
```

首先，导入一下常用的python库：

 *  argparse： 它是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息
 *  math： 调用这个库进行数学运算
 *  os： 它提供了多种操作系统的接口。通过os模块提供的操作系统接口，我们可以对操作系统里文件、终端、进程等进行操作
 *  random：是使用随机数的Python标准库。random库主要用于生成随机数
 *  sys： 它是与python解释器交互的一个接口，该模块提供对解释器使用或维护的一些变量的访问和获取，它提供了许多函数和变量来处理 Python 运行时环境的不同部分
 *  time： Python中处理时间的标准库，是最基础的时间处理库
 *  copy： Python 中赋值语句不复制对象，而是在目标和对象之间创建绑定 (bindings) 关系。copy模块提供了通用的浅层复制和深层复制操作
 *  datetime：是Python常用的一个库，主要用于时间解析和计算
 *  pathlib： 这个库提供了一种面向对象的方式来与文件系统交互，可以让代码更简洁、更易读

然后再导入一些pytorch库：

 *  numpy： 科学计算库，提供了矩阵，线性代数，傅立叶变换等等的解决方案, 最常用的是它的N维数组对象
 *  torch： 这是主要的Pytorch库。它提供了构建、训练和评估神经网络的工具
 *  torch.distributed： torch.distributed包提供Pytorch支持和通信基元，对多进程并行，在一个或多个机器上运行的若干个计算阶段
 *  torch.nn： torch下包含用于搭建神经网络的modules和可用于继承的类的一个子包
 *  yaml： yaml是一种直观的能够被电脑识别的的数据序列化格式，容易被人类阅读，并且容易和脚本语言交互。一般用于存储配置文件
 *  torch.cuda.amp： 自动混合精度训练 —— 节省显存并加快推理速度
 *  torch.nn.parallel： 构建分布式模型，并行加速程度更高，且支持多节点多gpu的硬件拓扑结构
 *  torch.optim： 优化器 Optimizer。主要是在模型训练阶段对模型可学习参数进行更新，常用优化器有 SGD，RMSprop，Adam等
 *  tqdm： 就是我们看到的训练时进度条显示

### 1.3 获取当前文件的绝对路径 

```java
'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()  # __file__指的是当前文件(即train.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/train.py
ROOT = FILE.parents[0]  # YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径
```

这段代码会获取当前文件的绝对路径，并使用Path库将其转换为Path对象。

这一部分的主要作用有两个：

 *  将当前项目添加到系统路径上，以使得项目中的模块可以调用。
 *  将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件。

### 1.4 加载自定义模块 

```java
'''===================3..加载自定义模块============================'''
import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
```

这些都是用户自定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换。具体来说，代码从如下几个文件中导入了部分函数和类：

 *  val： 这个是测试集，我们下一篇再具体讲
 *  models.experimental： 实验性质的代码，包括MixConv2d、跨层权重Sum等
 *  models.yolo： yolo的特定模块，包括BaseModel，DetectionModel，ClassificationModel，parse\_model等
 *  utils.autoanchor： 定义了自动生成锚框的方法
 *  utils.autobatch： 定义了自动生成批量大小的方法
 *  utils.callbacks： 定义了回调函数，主要为logger服务
 *  utils.datasets： dateset和dateloader定义代码
 *  utils.downloads： 谷歌云盘内容下载
 *  utils.general： 定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
 *  utils.loggers ： 日志打印
 *  utils.loss： 存放各种损失函数
 *  utils.metrics： 模型验证指标，包括ap，混淆矩阵等
 *  utils.plots.py： 定义了Annotator类，可以在图像上绘制矩形框和标注信息
 *  utils.torch\_utils.py： 定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等

通过导入这些模块，可以更方便地进行目标检测的相关任务，并且减少了代码的复杂度和冗余。

### 1.5 分布式训练初始化 

```java
'''================4.分布式训练初始化==========================='''
# https://pytorch.org/docs/stable/elastic/run.html该网址有详细介绍
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # -本地序号。这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))  # -进程序号。这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总共有几个 Worker
'''
   查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，
   若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配。
'''
```

接下来是设置分布式训练时所需的环境变量。分布式训练指的是多GPU训练，将训练参数分布在多个GPU上进行训练，有利于提升训练效率。

## 🚀二、执行main（）函数 

### 2.1 检查分布式训练环境 

```java
def main(opt, callbacks=Callbacks()):
    '''
    2.1  检查分布式训练环境
    '''
    # Checks
    if RANK in [-1, 0]:  # 若进程编号为-1或0
        # 输出所有训练参数 / 参数以彩色的方式表现
        print_args(FILE.stem, opt)
        # 检测YOLO v5的github仓库是否更新，若已更新，给出提示
        check_git_status()
        # 检查requirements.txt所需包是否都满足
        check_requirements(exclude=['thop'])
```

这段代码主要是检查分布式训练的环境。

若RANK为-1或0，会执行下面三行代码，打印参数并检查github仓库和依赖库。

 *  第一行代码，负责打印文件所用到的参数信息，这个参数包括命令行传入进去的参数以及默认参数
 *  第二行代码，检查yolov5的github仓库是否更新，如果更新的话，会有一个提示
 *  第三行代码，检查requirements中要求的安装包有没有正确安装成功，没有成功的话会给予一定的提示

### 2.2 判断是否断点续训 

```java
'''
    2.2  判断是否断点续训
    '''
    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        # isinstance()是否是已经知道的类型
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # 判断是否为文件，若不是文件抛出异常
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt.yaml是训练时的命令行参数文件
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            # 超参数替换，将训练时的命令行参数加载进opt参数对象中
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        # 打印从ckpt恢复断点训练信息
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        # 不使用断点续训，就从文件中读取相关参数
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        # 如果模型文件和权重文件为空，弹出警告
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 如果要进行超参数进化，重建保存路径
        if opt.evolve:
            # 设置新的项目输出目录
            opt.project = str(ROOT / 'runs/evolve')
            # 将resume传递给exist_ok
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        # 根据opt.project生成目录，并赋值给opt.save_dir  如: runs/train/exp1
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
```

这段代码主要是关于断点训练的判断和准备。

断点训练是当训练异常终止或想调节超参数时，系统会保留训练异常终止前的超参数与训练参数，当下次训练开始时，并不会从头开始，而是从上次中断的地方继续训练。

 *  使用断点续训，就从last.pt中读取相关参数
 *  不使用断点续训，就从文件中读取相关参数

### 2.3 判断是否分布式训练 

```java
'''
    2.3  判断是否分布式训练
    '''
    # DDP mode -->  支持多机多卡、分布式训练
    # 选择程序装载的位置
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 当进程内的GPU编号不为-1时，才会进入DDP
    if LOCAL_RANK != -1:
        #  用于DDP训练的GPU数量不足
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # WORLD_SIZE表示全局的进程数
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        # 不能使用图片采样策略
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        # 不能使用超参数进化
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        # 设置装载程序设备
        torch.cuda.set_device(LOCAL_RANK)
        # 保存装载程序的设备
        device = torch.device('cuda', LOCAL_RANK)
        # torch.distributed是用于多GPU训练的模块
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
```

这段代码主要是检查DDP训练的配置，并设置GPU。

DDP（Distributed Data Parallel）用于单机或多机的多GPU分布式训练，但目前DDP只能在Linux下使用。这部分它会选择你是使用cpu还是gpu，假如你采用的是分布式训练的话，它就会额外执行下面的一些操作，我们这里一般不会用到分布式，所以也就没有执行什么东西。

### 2.4 判断是否进化训练 

```java
'''
    2.4  判断是否进化训练
    '''
    # Train 训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:# 如果不使用超参数进化
        # 开始训练
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            # 如果全局进程数大于1并且RANK等于0
            # 日志输出 销毁进程组
            LOGGER.info('Destroying process group... ')
            # 训练完毕，销毁所有进程
            dist.destroy_process_group()
```

这段代码是不进行进化训练的情况，此时正常训练。

如果输入evolve会执行else下面这些代码，因为我们没有输入evolve并且不是分布式训练，因此会执行train函数。也就是说，当不使用超参数进化训练时，直接把命令行参数传入train函数，训练完成后销毁所有进程。

接下来我们再看看使用超参数进化训练的情况：

```java
# Evolve hyperparameters (optional) 遗传进化算法，边进化边训练
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数列表(突变范围 - 最小值 - 最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)
        # 加载默认超参数
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

            """
            遗传算法调参：遵循适者生存、优胜劣汰的法则，即寻优过程中保留有用的，去除无用的。
            遗传算法需要提前设置4个参数: 群体大小/进化代数/交叉概率/变异概率
            """
```

这段代码是使用超参数进化训练的前期准备

首先指定每个超参数的突变范围、最大值、最小值，再为超参数的结果保存做好准备。

```java
# 选择超参数的遗传迭代次数 默认为迭代300次
        for _ in range(opt.evolve):  # generations to evolve
            # 如果evolve.csv文件存在
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                # 选择超参进化方式，只用single和weighted两种
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 选取至多前五次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                # fitness()为x前四项加权 [P, R, mAP@0.5, mAP@0.5:0.95]
                # np.argsort只能从小到大排序, 添加负号实现从大到小排序, 算是排序的一个代码技巧
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据(mp, mr, map50, map)的加权和来作为权重计算hyp权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # 根据权重的几率随机挑选适应度历史前5的其中一个
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    # 对hyp乘上对应的权重融合层一个hpy, 再取平均(除以权重和)
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate 突变（超参数进化）
                mp, s = 0.8, 0.2  # mutation probability, sigma：突变概率
                npr = np.random
                # 根据时间设置随机数种子
                npr.seed(int(time.time()))
                # 获取突变初始值, 也就是meta三个值的第一个数据
                # 三个数值分别对应着: 变异初始概率, 最低限值, 最大限值(mutation scale 0-1, lower_limit, upper_limit)
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                # 确保至少其中有一个超参变异了
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits 限制hyp在规定范围内
            for k, v in meta.items():
                # 这里的hyp是超参数配置文件对象
                # 而这里的k和v是在元超参数中遍历出来的
                # hyp的v是一个数，而元超参数的v是一个元组
                hyp[k] = max(hyp[k], v[1])  # 先限定最小值，选择二者之间的大值 ，这一步是为了防止hyp中的值过小
                hyp[k] = min(hyp[k], v[2])  # 再限定最大值，选择二者之间的小值
                hyp[k] = round(hyp[k], 5)  # 四舍五入到小数点后五位
                # 最后的值应该是 hyp中的值与 meta的最大值之间的较小者

            # Train mutation 使用突变后的参超，测试其效果
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            # 将结果写入results，并将对应的hyp写到evolve.txt，evolve.txt中每一行为一次进化的结果
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            # 保存hyp到yaml文件
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results 将结果可视化 / 输出保存信息
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')
```

这段代码是开始超参数进化训练。

超参数进化的步骤如下：

 *  1.若存在evolve.csv文件，读取文件中的训练数据，选择超参进化方式，结果最优的训练数据突变超参数
 *  2.限制超参进化参数hyp在规定范围内
 *  3.使用突变后的超参数进行训练，测试其效果
 *  4.训练结束后，将训练结果可视化，输出保存信息保存至evolution.csv，用于下一次的超参数突变。

原理：根据生物进化，优胜劣汰，适者生存的原则，每次迭代都会保存更优秀的结果，直至迭代结束。最后的结果即为最优的超参数

注意：使用超参数进化时要经过至少300次迭代，每次迭代都会经过一次完整的训练。因此超参数进化及其耗时，大家需要根据自己需求慎用。

## 🚀三、设置opt参数 

```java
=============================================三、设置opt参数==================================================='''

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 预训练权重文件
    parser.add_argument('--weights', type=str, default=ROOT / 'pretrained/yolov5s.pt', help='initial weights path')
    # 训练模型
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    # 训练路径，包括训练集，验证集，测试集的路径，类别总数等
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire_data.yaml', help='dataset.yaml path')
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # epochs: 训练轮次， 默认轮次为300次
    parser.add_argument('--epochs', type=int, default=300)
    # batchsize: 训练批次， 默认bs=16
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    # imagesize: 设置图片大小, 默认640*640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # rect: 是否采用矩形训练，默认为False
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # resume: 是否接着上次的训练结果，继续训练
    # 矩形训练：将比例相近的图片放在一个batch（由于batch里面的图片shape是一样的）
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # nosave: 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # best.pt/ last.pt 不建议运行代码添加 --nosave
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # noval: 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # noautoanchor: 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # evolve: 参数进化， 遗传算法调参
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # bucket: 谷歌优盘 / 一般用不到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # cache: 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # mage-weights: 使用图片采样策略，默认不使用
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # device: 设备选择
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # multi-scale 是否进行多尺度训练
    parser.add_argument('--multi-scale', default=True, help='vary img-size +/- 50%%')
    # single-cls: 数据集是否多类/默认True
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # optimizer: 优化器选择 / 提供了三种优化器
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # sync-bn: 是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # dataloader的最大worker数量 （使用多线程加载图片）
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    # 训练结果的保存路径
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # 训练结果的文件名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 项目位置是否存在 / 默认是都不存在
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 四元数据加载器: 允许在较低 --img 尺寸下进行更高 --img 尺寸训练的一些好处。
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # cos-lr: 余弦学习率
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # 早停止耐心次数 / 100次不更新就停止训练
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # --freeze冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    # --save-period 多少个epoch保存一下checkpoint
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # --local_rank 进程编号 / 多卡使用
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # Weights & Biases arguments
    # 在线可视化工具，类似于tensorboard工具
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    # 使用数据的版本
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```

opt参数解析：

 *  cfg:  模型配置文件，网络结构
 *  data: 数据集配置文件，数据集路径，类名等
 *  hyp:  超参数文件
 *  epochs: 训练总轮次
 *  batch-size:  批次大小
 *  img-size:  输入图片分辨率大小
 *  rect:  是否采用矩形训练，默认False
 *  resume: 接着打断训练上次的结果接着训练
 *  nosave:  不保存模型，默认False
 *  notest:  不进行test，默认False
 *  noautoanchor: 不自动调整anchor，默认False
 *  evolve:  是否进行超参数进化，默认False
 *  bucket:  谷歌云盘bucket，一般不会用到
 *  cache-images:  是否提前缓存图片到内存，以加快训练速度，默认False
 *  weights:  加载的权重文件
 *  name:  数据集名字，如果设置：results.txt to results\_name.txt，默认无
 *  device:  训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
 *  multi-scale:  是否进行多尺度训练，默认False
 *  single-cls:  数据集是否只有一个类别，默认False
 *  adam:  是否使用adam优化器
 *  sync-bn:  是否使用跨卡同步BN,在DDP模式使用
 *  local\_rank:  gpu编号
 *  logdir:  存放日志的目录 
 *  workers: dataloader的最大worker数量

（关于调参，推荐大家看@迪菲赫尔曼大佬的这篇文章：[手把手带你调参YOLOv5 (v5.0-v7.0)（训练）][YOLOv5 _v5.0-v7.0]）

## 🚀四、执行train（）函数 

### 4.1 加载参数和初始化配置信息 

#### 4.1.1 载入参数 

```java
''' =====================1.载入参数和初始化配置信息==========================  '''
    '''
    1.1 载入参数
    '''
def train(hyp,  # 超参数 可以是超参数配置文件的路径或超参数字典 path/to/hyp.yaml or hyp
          opt,  # main中opt参数
          device,  # 当前设备
          callbacks  # 用于存储Loggers日志记录器中的函数，方便在每个训练阶段控制日志的记录情况
          ):
    # 从opt获取参数。日志保存路径，轮次、批次、权重、进程序号(主要用于分布式训练)等
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
```

这段代码是接收传来的参数

 *  hyp： 超参数，不使用超参数进化的前提下也可以从opt中获取
 *  opt： 全部的命令行参数
 *  device： 指的是装载程序的设备
 *  callbacks： 指的是训练过程中产生的一些参数

#### 4.1.2 创建训练权重目录和保存路径 

```java
'''
    1.2 创建训练权重目录，设置模型、txt等保存的路径
    '''
    # Directories 获取记录训练日志的保存路径
    # 设置保存权重路径 如runs/train/exp1/weights
    w = save_dir / 'weights'  # weights dir
    # 新建文件夹 weights train evolve
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 保存训练结果的目录，如last.pt和best.pt
    last, best = w / 'last.pt', w / 'best.pt'
```

这段代码主要是创建权重文件保存路径，权重名字和训练日志txt文件

每次训练结束后，系统会产生两个模型，一个是last.pt，一个是best.pt。顾名思义，last.pt即为训练最后一轮产生的模型，而best.pt是训练过程中，效果最好的模型。

然后创建文件夹，保存训练结果的模型文件路径 以及验证集输出结果的txt文件路径，包含迭代的次数，占用显存大小，图片尺寸，精确率，[召回率][Link 4]，位置损失，类别损失，置信度损失和map等。

#### 4.1.3 读取超参数配置文件 

```java
'''
    1.3 读取hyp(超参数)配置文件
    '''
    # Hyperparameters 加载超参数
    if isinstance(hyp, str): # isinstance()是否是已知类型。 判断hyp是字典还是字符串
        # 若hyp是字符串，即认定为路径，则加载超参数为字典
        with open(hyp, errors='ignore') as f:
            # 加载yaml文件
            hyp = yaml.safe_load(f)  # load hyps dict 加载超参信息
    # 打印超参数 彩色字体
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
```

这段代码主要是加载一些训练过程中需要使用的超参数，并打印出来

首先，检查超参数是字典还是字符串，若为字符串，则认定为.yaml文件路径，再将yaml文件加载为字典。这里导致超参数的数据类型不同的原因是，超参数进化时，传入train()函数的超参数即为字典。而从命令行参数中读取的则为文件路径。

然后将打印这些超参数。

#### 4.1.4 设置参数的保存路径 

```java
'''
    1.4 设置参数的保存路径
    '''
    # Save run settings 保存训练中的参数hyp和opt
    with open(save_dir / 'hyp.yaml', 'w') as f:
        # 保存超参数为yaml配置文件
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        # 保存命令行参数为yaml配置文件
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        # 定义数据集字典
    data_dict = None
```

这段代码主要是将训练的相关参数全部写入 

将本次运行的超参数(hyp)和选项操作(opt)给保存成yaml格式，保存在了每次训练得到的exp文件中，这两个yaml显示了我们本次训练所选择的hyp超参数和opt参数。

还有一点，yaml.safe\_load(f)是加载yaml的标准函数接口，保存超参数为yaml配置文件。 yaml.safe\_dump()是将yaml文件序列化，保存命令行参数为yaml配置文件。  
`vars(opt)`的作用是把数据类型是Namespace的数据转换为字典的形式。

#### 4.1.5 加载日志信息 

```java
'''
    1.5 加载相关日志功能:如tensorboard,logger,wandb
    '''
    # Loggers 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    if RANK in [-1, 0]:  # 如果进程编号为-1或0
        # 初始化日志记录器实例
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # W&B # wandb为可视化参数工具
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            # 如果使用中断训练 再读取一次参数
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            # 将日志记录器中的方法与字符串进行绑定
            callbacks.register_action(k, callback=getattr(loggers, k))
```

这段代码主要是打印日志信息(logger + wandb) 

首先设置wandb和tb两种日志，并开始初始化日志记录器实例。

然后将日志记录器中的函数记录到callbacks内，方便在训练的不同阶段，利用callbacks.run()函数对日志的记录做统一处理。

在日志文件中，基于wandb与tensorboard这两个库来完成训练过程中的可视化操作。在这个文件中完成对于程序训练日志的记录过程。

#### 4.1.6 加载其它参数 

```java
'''
    1.6 配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Config 画图
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    # 设置随机种子
    init_seeds(1 + RANK)
    # 加载数据配置信息
    with torch_distributed_zero_first(LOCAL_RANK): # torch_distributed_zero_first 同步所有进程
        data_dict = data_dict or check_dataset(data)  # check if None  check_dataset 检查数据集，如果没找到数据集则下载数据集(仅适用于项目中自带的yaml文件数据集)
    # 获取训练集、测试集图片路径
    train_path, val_path = data_dict['train'], data_dict['val']
    # nc：数据集有多少种类别
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # names: 数据集所有类别的名字，如果设置了single_cls则为一类
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # 判断类别长度和文件是否对应
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
```

这段代码主要作用是做一些变量的配置

首先根据plots的true或者false来判断是否将训练过程及结果给画出来，紧接着判断电脑是否支持cuda。

然后设置随机种子，下一行的torch\_distributed\_zero\_first(LOCAL\_RANK)与分布式训练相关的，如果不进行分布式训练则不执行，利用check\_dataset会进行数据集检查读取操作，获取训练集和测试集图片路径。

接着利用nc获取数据集的种类，names会进行类的种数以及类的名称是否相同的判断，不相同会进行报错处理，然后保存类别数量以及类别名，并完成检查。

最后会进行是否是coco数据集进行判断，如果是的话会进行一些额外的操作，如果不是，则输出false。

> 4.1 加载参数和初始化配置信息小结：
> 
> 解析各种yaml的参数＋创建训练权重目录和保存路径+ 读取超参数配置文件 + 设置保存参数保存路径 + 加载数据配置信息 + 加载日志信息(logger + wandb) + 加载其他参数(plots、cuda、nc、names、is\_coco)

### 4.2 加载网络模型 

#### 4.2.1 加载预训练模型 

```java
''' =====================2.model：加载网络模型==========================  '''
    # Model 载入模型
    # 检查文件后缀是否是.pt
    check_suffix(weights, '.pt')  # check weights
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，可以根据自己的模型选择预训练权重
    pretrained = weights.endswith('.pt')

    '''
    2.1预训练模型加载 
    '''
    if pretrained:
        # 使用预训练的话：
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果本地不存在就从google云盘中自动下载模型
            # 通常会下载失败，建议提前下载下来放进weights目录
            weights = attempt_download(weights)  # download if not found locally
        # ============加载模型以及参数================= #
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        """
        两种加载模型的方式: opt.cfg / ckpt['model'].yaml
        这两种方式的区别：区别在于是否是使用resume
        如果使用resume-断点训练: 
        将opt.cfg设为空，选择ckpt['model']yaml创建模型, 且不加载anchor。
        这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        原因：
        使用断点训练时,保存的模型会保存anchor,所以不需要加载，
        主要是预训练权重里面保存了默认coco数据集对应的anchor，
        如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor。
        """
        # ***加载模型*** #
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

        # ***以下三行是获得anchor*** #
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect

        # ***模型创建*** #
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为ture 则会少加载两个键对（anchors, anchor_grid）
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # #直接加载模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
```

这段代码主要是加载模型，分为使用预训练权重参数文件与不使用预训练权重参数文件。

首先会去检测传进来的权重参数后缀名是否以.pt结尾，分两类：

 *  如果传入权重文件，直接model.load\_state\_dict加载模型
 *  如果没有传入权重文件，就回去会尝试去yolov5官方仓库去下载权重文件，加载权重文件，紧接着会根据你的权重文件中会带着一个yolov5s.yaml文件，代码根据yolov5s.yaml进行模型的训练。（通俗的理解就是我们预训练模型是yolov5s.pt，我们的新模型是基于我们自己的识别检测需求在yolov5s的基础上完成的。）

最后，获取的train\_path和test\_path分别表示在data.yaml中训练数据集和测试数据集的地址。

> 这里使用预训练权重参数，是类似于迁移学习。
> 
> 预训练的模型是检测coco数据集的模型，数据集中有80个类别，而自己的训练集类别以及类别的数量，并不与coco数据集相同。所以要先加载一个新的模型，把预训练的参数加载至模型作为初始参数，再把识别的类别改成自己的数据集要识别的类别。接下来将预训练参数中与新模型中相同的参数加载至模型。

#### 4.2.2 设置冻结层 

```java
'''
    2.2 冻结层
    '''
    # Freeze 冻结训练的网络层
    """
    冻结模型层,设置冻结层名字即可
    作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    freeze为命令行参数，默认为0，表示不冻结
    """
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    # 首先遍历所有层
    for k, v in model.named_parameters():
        # 为所有层的参数设置梯度
        v.requires_grad = True  # train all layers
        # 判断是否需要冻结
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            # 冻结训练的层梯度不更新
            v.requires_grad = False

    # Image size 设置训练和测试图片尺寸
    # 获取模型总步长和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 检查输入图片分辨率是否能被32整除
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size 设置一次训练所选取的样本数
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
       # 确保batch size满足要求
        batch_size = check_train_batch_size(model, imgsz)
```

这段代码是设置冻结层的，即将模型的部分权重冻结，在模型训练过程中不会变化，只训练冻结层以为的权重参数 

冻结层的原理是通过设置每个层参数中的requires\_grad属性实现的。

 *  若require\_grad为True，在反向传播时就会求出此tensor的梯度
 *  若require\_grad为False，则不会求该tensor的梯度。冻结就是通过对某些层不求梯度实现的。默认不进行参数冻结

通过Freeze这部分代码，我们可以手动去控制想冻结哪些层。但是作者这里列出来这部分代码的目的其实并不是鼓励使用冻结指定层，因为作者认为这样效果其实并不是很好。

#### 4.2.3 设置优化器 

```java
'''
    2.3 优化器设置
    '''
    # Optimizer 优化器
    nbs = 64  # nominal batch size
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # 打印缩放后的权重衰减超参数
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
```

这段代码是参数设置(nbs、accumulate、hyp\[‘weight\_decay’\]) 

nbs指的是nominal batch size，名义上的batch\_size。这里的nbs跟命令行参数中的batch\_size不同，命令行中的batch\_size默认为16，nbs设置为64。

accumulate 为累计次数，在这里 nbs/batch\_size（64/16）计算出 opt.batch\_size输入多少批才达到nbs的水平。简单来说，nbs为64，代表想要达到的batch\_size，这里的数值是64；batch\_size为opt.batch\_size，这里的数值是16。64/16等于4，也就是opt.batch\_size需要输入4批才能达到nbs，accumulate等于4。(round表示四舍五入取整数，而max表示accumulate不能低于1。)

当给模型喂了4批图片数据后，将四批图片数据得到的梯度值，做累积。当每累积到4批数据时，才会对参数做更新，这样就实现了与batch\_size=64时相同的效果。

最后还要做权重参数的缩放，因为batch\_size发生了变化，所有权重参数也要做相应的缩放。

```java
# 将模型分成三组（BN层的weight，卷积层的weights，biases）进行优化
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    # 遍历网络中的所有层，每遍历完一层向更深的层遍历
    for v in model.modules():
        # hasattr: 测试指定的对象是否具有给定的属性，返回一个布尔值
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            # 将层的bias添加至g2
            g2.append(v.bias)
        # YOLO v5的模型架构中只有卷积层和BN层
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            # 将BN层的权重添加至g0 未经过权重衰减
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            # 将层的weight添加至g1 经过了权重衰减
            # 这里指的是卷积层的weight
            g1.append(v.weight)
```

这段代码是分组优化(g0、g1、g2) 

将模型的参数分为三组，g0表示归一化层中的所有权重参数，g1表示卷积层中所有的权重参数，g2表示所有的偏置参数。

model.modules()迭代遍历模型的所有子层，（后期改进为model.named\_modules()不但返回模型的所有子层，还会返回这些层的名字。）

hasattr（）函数来判断遍历的每个层对象是否拥有相对应的属性，将所有参数分成三类：weight、bn, bias。

```java
# 选用优化器，并设置g0(bn参数)组的优化方式
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 将卷积层的参数添加至优化器 并做权重衰减
    # add_param_group()函数为添加一个参数组，同一个优化器可以更新很多个参数组，不同的参数组可以设置不同的超参数
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # 将所有的bias添加至优化器
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # 打印优化信息
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    # 在内存中删除g0 g1 g2 目的是节省空间
    del g0, g1, g2
```

这段代码是选择优化器，然后为三个优化器选择优化方式，最后删除变量

首先判断是否使用adam优化器，初始参数为批归一化层中的参数。如果不使用adam优化器，则直接使用SGD随机梯度下降。

然后将g1(卷积层中的权重参数)，g2(偏置参数)，添加进优化器。 add\_param\_group()函数可以为优化器中添加一个参数组。一个优化器可以更新多个参数组，不同的参数组可以使用不同的超参数。

#### 4.2.4 设置学习率 

```java
'''
    2.4 学习率设置
    '''
    # Scheduler  设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    if opt.linear_lr:
        # 使用线性学习率
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 使用余弦退火学习率
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # 可视化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
```

这段代码主要是设置学习率衰减方式

在训练过程中变更学习率可能会让训练效果更好，YOLOv5提供了两种学习率变化的策略：

 *  一种是linear\_lr（线性学习率），是通过线性插值的方式调整学习率
 *  另一种则是One Cycle（余弦退火学习率），即周期性学习率调整中，周期被设置为1。在一周期策略中，最大学习率被设置为 LR Range test 中可以找到的最高值，最小学习率比最大学习率小几个数量级。这里默认one\_cycle。

#### 4.2.5 训练前最后准备 

```java
'''
    2.5 训练前最后准备
    '''
    # EMA 设置ema（指数移动平均），考虑历史值对参数的影响，目的是为了收敛的曲线更加平滑
    ema = ModelEMA(model) if RANK in [-1, 0] else None # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建

    # Resume 断点续训
    # 断点续训其实就是把上次训练结束的模型作为预训练模型，并从中加载参数
    start_epoch, best_fitness = 0, 0.0
    if pretrained:# 如果有预训练
        # Optimizer 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            # 将预训练模型中的参数加载进优化器
            optimizer.load_state_dict(ckpt['optimizer'])
            # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
            # 获取预训练模型中的最佳fitness，保存为best.pt
            best_fitness = ckpt['best_fitness']

        # EMA
        # 加载ema模型和updates参数,保持ema的平滑性,现在yolov5是ema和model都保存了
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs 加载训练的迭代次数
        start_epoch = ckpt['epoch'] + 1 # 从上次的epoch接着训练
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        # 如果训练的轮数小于开始的轮数
        if epochs < start_epoch:
            # 打印日志恢复训练
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            # 计算新的轮数
            epochs += ckpt['epoch']  # finetune additional epochs
        # 将预训练的相关参数从内存中删除
        del ckpt, csd

    # DP mode 使用单机多卡模式训练，目前一般不使用
    # rank为进程编号。如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  多卡归一化
    if opt.sync_bn and cuda and RANK != -1:# 多卡训练，把不同卡的数据做个同步
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
```

这段代码主要是训练前最后的准备工作（EMA ＋断点续训+ 迭代次数的加载 + DP ＋SyncBatchNorm）

 *  EMA为指数加权平均或滑动平均。其将前面模型训练权重，偏差进行保存，在本次训练过程中，假设为第n次，将第一次到第n-1次以指数权重进行加和，再加上本次的结果，且越远离第n次，指数系数越大，其所占的比重越小。
 *  断点续训。可以理解为把上次中断结束时的模型，作为新的预训练模型，然后从中获取上次训练时的参数，并恢复训练状态。
 *  epoch迭代次数。1个epoch等于使用训练集中的全部样本训练一次，epoch的大小跟迭代次数有着密切的关系，通常在迭代次数处于2000-3000之间损失已经处于平稳。
 *  DP mode。DataParallel单机多卡模式自动将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总。值得注意的是，模型和数据都需要先导入进 GPU 中，DataParallel 的 module 才能对其进行处理，否则会报错。
 *  SyncBatchNorm。SyncBatchNorm主要用于解决多卡归一化同步问题，每张卡单独计算均值，然后同步，得到全局均值。用全局均值计算每张卡的方差，然后同步即可得到全局方差，但两次会消耗时间挺长。

> 4.2 加载网络模型小结：
> 
> （1）载入模型：载入模型(预训练/不预训练) + 检查数据集 + 设置数据集路径参数(train\_path、test\_path) + 设置冻结层
> 
> （2）优化器：参数设置(nbs、accumulate、hyp\[‘weight\_decay’\]) + 分组优化(pg0、pg1、pg2) + 选择优化器 + 为三个优化器选择优化方式 + 删除变量
> 
> （3）学习率：线性学习率 + one cycle学习率 + 实例化 scheduler + 画出学习率变化曲线
> 
> （4）训练前最后准备：EMA ＋断点续训+ 迭代次数的加载 + DP ＋SyncBatchNorm）

### 4.3 加载数据集 

#### 4.3.1 创建数据集 

```java
''' =====================3.加载训练数据集==========================  '''
    '''
    3.1 创建数据集
    '''
    # Trainloader 训练集数据加载
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    '''
      返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    # 标签编号最大值
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    # 类别总数
    nb = len(train_loader)  # number of batches
    # 如果小于类别数则表示有问题
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0 验证集数据集加载
    if RANK in [-1, 0]:# 加载验证集数据加载器
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:# 没有使用resume
            # 统计dataset的label信息
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:# plots画出标签信息
                plot_labels(labels, names, save_dir)
```

这段代码主要是创建训练用的数据集

首先，通过create\_dataloader()函数得到两个对象。一个为train\_loader，另一个为dataset。

 *  train\_loader为训练数据加载器，可以通过for循环遍历出每个batch的训练数据
 *  dataset为数据集对象，包括所有训练图片的路径，所有标签，每张图片的大小，图片的配置，超参数等等

然后将所有样本的标签拼接到一起，统计后做可视化，同时获得所有样本的类别，根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化。

#### 4.3.2 计算anchor 

```java
'''
  3.2 计算anchor
  '''
            # Anchors 计算默认锚框anchor与数据集标签框的高宽比
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                '''
                参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
            # 半进度
            model.half().float()  # pre-reduce anchor precision
        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.run('on_pretrain_routine_end')

    # DDP mode 如果rank不等于-1,则使用DistributedDataParallel模式
    if cuda and RANK != -1:
        # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
```

这段代码主要是计算默认锚点anchor与数据集标签框的长宽比值

check\_anchors计算默认锚点anchor与数据集标签框的长宽比值，标签的长h宽w与anchor的长h\_a宽w\_a的比值, 即h/h\_a, w/w\_a都要在(1/hyp\[‘anchor\_t’\], hyp\[‘anchor\_t’\])是可以接受的，如果标签框满足上面条件的数量小于总数的99%，则根据k-mean算法聚类新的锚点anchor。

> 4.3 加载数据集小结：
> 
> 加载训练集dataloader、dataset + 参数(mlc、nb) + 加载验证集testloader + 设置labels相关参数(labels、c) ＋plots可视化数据集labels信息＋检查anchors(k-means + 遗传进化算法)＋model半精度

### 4.4 训练过程 

#### 4.4.1 初始化训练需要的模型参数 

```java
''' =====================4.训练==========================  '''

    '''
    4.1 初始化训练需要的模型参数
    '''
    # Model attributes  根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # box为预测框的损失
    hyp['box'] *= 3 / nl  # scale to layers
    # cls为分类的损失
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # obj为置信度损失
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # 标签平滑
    hyp['label_smoothing'] = opt.label_smoothing
    # 设置模型的类别，然后将检测的类别个数保存到模型
    model.nc = nc  # attach number of classes to model
    # 设置模型的超参数，然后将超参数保存到模型
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练的样本标签得到类别权重，然后将类别权重保存至模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字，然后将分类标签保存至模型
    model.names = names
```

这段代码主要是根据自己数据集的类别数设置分类损失的系数，位置损失的系数。设置类别数，超参数等操作

其中，

 *  box： 预测框的损失
 *  cls： 分类的损失
 *  obj： 置信度损失
 *  label\_smoothing :标签平滑

#### 4.4.2 训练热身部分 

```java
'''
    4.2 训练热身部分
    '''
    # Start training
    t0 = time.time() # 获取当前时间
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化 map和result
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练    GradScaler + autocast
    scaler = amp.GradScaler(enabled=cuda)
    # 早停止，不更新结束训练
    stopper = EarlyStopping(patience=opt.patience)
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    # 打印日志输出信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n' # 打印训练和测试输入图片分辨率
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n' # 加载图片时调用的cpu进程数
                f"Logging results to {colorstr('bold', save_dir)}\n" # 日志目录
                f'Starting training for {epochs} epochs...') # 从哪个epoch开始训练
```

这段代码是训练前的热身准备，做一些参数的初始化

这里要提到两个点：

第一个是warmup。warmup是一种学习率的优化方法，最早出现在ResNet的论文中。简单来说，在模型刚开始训练时，使用较小的学习率开始摸索，经过几轮迭代后使用大的学习率加速收敛，在快接近目标时，再使用小学习率，避免错过目标。

第二个是早停机制。当训练一定的轮数后，如果模型效果未提升，就让模型提前停止训练。这里的默认轮数为100轮，判断模型的效果为fitness，fitness为0.1乘mAP@0.5加上0.9乘mAP@0.5:0.95。

#### 4.4.3 开始训练 

```java
'''
    4.3 开始训练
    '''
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        '''
        告诉模型现在是训练阶段 因为BN层、DropOut层、两阶段目标检测模型等
        训练阶段阶段和预测阶段进行的运算是不同的，所以要将二者分开
        model.eval()指的是预测推断阶段
        '''
        model.train()

        # Update image weights (optional, single-GPU only)  更新图片的权重
        if opt.image_weights: # 获取图片采样的权重
            # 经过一轮训练，若哪一类的不精确度高，那么这个类就会被分配一个较高的权重，来增加它被采样的概率
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 将计算出的权重换算到图片的维度，将类别的权重换算为图片的权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # 通过random.choices生成图片索引indices从而进行采样，这时图像会包含一些难识别的样本
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
```

这段代码主要是释放训练开始命令和更新权重

首先训练过程走起，通过model.train()函数告诉模型已经进入了训练阶段。因为有些层或模型在训练阶段与预测阶段进行的操作是不一样的，所以要通过model.train()函数用来声明，接下来是训练。

然后是更新图片的权重。训练时有些类的准确率可能比较难以识别，准确率并不会很高。在更新图片权重时就会把这些难以识别的类挑出来，并为这个类产生一些权重高的图片，以这种方式来增加识别率低的类别的数据量。提高准确率。

```java
mloss = torch.zeros(3, device=device)  # mean losses
        # 分布式训练的设置
        # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        # 将训练数据迭代器做枚举，可以遍历出索引值
        pbar = enumerate(train_loader)
        # 训练参数的表头
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        if RANK in [-1, 0]:
            # 通过tqdm创建进度条，方便训练信息的展示
            pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # 将优化器中的所有参数梯度设为0
        optimizer.zero_grad()
```

这段代码主要是分布式训练的设置，以及训练时控制台的显示

首先DDP模式打乱数据，并进行随机采样。

然后设置训练时控制台的显示。LOGGER.info是输出的表头， tqdm 显示进度条效果

最后将优化器中所有的参数梯度设为0

```java
for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图片加载至设备 并做归一化
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup 热身训练
            '''
            热身训练(前nw次迭代),热身训练迭代的次数iteration范围[1:nw] 
            在前nw次迭代中, 根据以下方式选取accumulate和学习率
            '''
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 遍历优化器中的所有参数组
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale: # 随机改变图片的尺寸
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
```

这段代码主要是分批加载数据和热身训练

首先分批加载训练数据，用ni计算当前迭代的次数，并作图片的归一化。

然后进行热身训练（warmup），这里只对训练初期使用较小的学习率。对于bias参数组的学习率策略是从0.1逐渐降低至初始学习率，其余参数组则从0开始逐渐增长至初始学习率。

最后进行多尺度训练：

 *  imgz： 默认训练尺寸
 *  gs： 模型最大stride=32

```java
# Forward 前向传播
            with amp.autocast(enabled=cuda):
                # 将图片送入网络得到一个预测结果
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练,平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward 反向传播 scale为使用自动混合精度运算
            scaler.scale(loss).backward()

            # Optimize 模型会对多批数据进行累积，只有达到累计次数的时候才会更新参数，再还没有达到累积次数时 loss会不断的叠加 不会被新的反传替代
            if ni - last_opt_step >= accumulate:
                '''
                 scaler.step()首先把梯度的值unscale回来，
                 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                 否则，忽略step调用，从而保证权重不更新（不被破坏）
                '''
                scaler.step(optimizer)  # optimizer.step 参数更新
                # 更新参数
                scaler.update()
                # 完成一次累积后，再将梯度清零，方便下一次清零
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                # 计数
                last_opt_step = ni
```

这段代码主要是正向传播、反向传播、以及更新参数。

首先正向传播即将图片输入模型，并做一次正向传播，最后得到一个结果。这个结果在训练初期的效果可能会比较差，将这个结果与图片的标签值求损失，目的就是让这个损失越来越小。

接下来将这个误差，通过链式求导法则，反向传播回每一层，求出每层的梯度。

最后利用optimizer.step更新参数。但是要注意，在更新参数时这里有一个不一样的地方，并不会在每次反向传播时更新参数，而是做一定的累积，反向传播的结果并不会顶替上一次反向传播结果，而是做一个累积。完成一次积累后，再将梯度清零，方便下一次清零。这样做是为了以更小的batch\_size实现更高的batch\_size效果。

```java
# Log 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in [-1, 0]:
                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # 计算显存
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # 进度条显示以上信息
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # 调用Loggers中的on_train_batch_end方法，将日志记录并生成一些记录的图片
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # 根据前面设置的学习率更新策略更新学习率
        scheduler.step()
```

这段代码主要是打印训练相关信息，结束后做权重衰减

首先将每批最后的数据输出至控制台。到此每批循环体结束。

然后在每所有批训练结束时，做权重衰减，进入下一轮的训练。

#### 4.4.4 训练完成保存模型 

```java
'''
        4.4 训练完成保存模型  
        '''
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                """
                测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                       results: [1] Precision 所有类别的平均precision(最大f1时)
                                [1] Recall 所有类别的平均recall
                                [1] map@0.5 所有类别的平均mAP@0.5
                                [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                                [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                       maps: [80] 所有类别的mAP@0.5:0.95
                """
                results, maps, _ = val.run(data_dict, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                                           batch_size=batch_size // WORLD_SIZE * 2, # 要保证batch_size能整除卡数
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls, # 是否是单类数据集
                                           dataloader=val_loader,
                                           save_dir=save_dir,  # 保存地址 runs/train/expn
                                           plots=False, # 是否可视化
                                           callbacks=callbacks,
                                           compute_loss=compute_loss) # 损失函数(train)

            # Update best mAP 更新best_fitness
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # 若当前的fitness大于最佳的fitness
            if fi > best_fitness:
                # 将最佳fitness更新为当前fitness
                best_fitness = fi
            # 保存验证结果
            log_vals = list(mloss) + list(results) + lr
            # 记录验证数据
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
```

这段代码主要是得到results, mAps相关信息

首先判断是否应当结束训练，若选择每轮验证或当前已是最后一轮的情况下，做一次验证。

然后计算出最好的模型。这里“最好”的评判标准即为fitness。fi: \[P, R, mAP@.5, mAP@.5-.95\]的一个加权值 = 0.1\*mAP@.5 + 0.9\*mAP@.5-.95，在评判标准中，更加强调mAP@0.5:0.95的作用。mAP@0.5:0.95大代表模型在多个IOU阈值的情况下，都可以较好的识别物体。

```java
# Save model 保存模型
            """
            保存带checkpoint的模型用于inference或resuming training
            保存模型, 还保存了epoch, results, optimizer等信息
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 将当前训练过程中的所有参数赋值给ckpt
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete 保存每轮的模型
                torch.save(ckpt, last)
                # 如果这个模型的fitness是最佳的
                if best_fitness == fi:
                    # 保存这个最佳的模型
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                # 模型保存完毕 将变量从内存中删除
                del ckpt
                # 记录保存模型时的日志
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU 停止单卡训练
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
```

这段代码主要是保存模型

首先将当前训练过程中的所有参数赋值给ckpt。

然后判断这个模型的fitness是否是最佳，如果是，就保存这个最佳模型，保存完毕将变量从内存中删除。

至此，训练结束（哎呀妈呀，可算结束了）

> 4.4 训练过程小结：
> 
> （1）初始化训练需要的模型参数：设置/初始化一些训练要用的参数(hyp\[‘box’\]、hyp\[‘cls’\]、hyp\[‘obj’\]、hyp\[‘label\_smoothing’\]）＋从训练样本标签得到类别权重model.class\_weights、model.names。
> 
> （2）热身部分：热身迭代的次数iterationsnw、last\_opt\_step、初始化maps和results、学习率衰减所进行到的轮次scheduler.last\_epoch + 设置amp混合精度训练scaler + 初始化损失函数compute\_loss + 打印日志信息)
> 
> （3）开始训练：图片采样策略 + Warmup热身训练 + multi\_scale多尺度训练 + amp混合精度训练 + accumulate 梯度更新策略+ 打印训练相关信息(包括当前epoch、显存、损失(box、obj、cls、total)＋当前batch的target的数量和图片的size等 + 调整学习率、scheduler.step() 、emp val.run()得到results, maps相关信息
> 
> （4）训练完成保存模型：将测试结果results写入result.txt中、wandb\_logger、Update best mAP 以加权mAP fitness为衡量标准＋保存模型

### 4.5 打印信息并释放显存 

```java
'''
    4.5 打印信息并释放显存 
    '''
    # 打印一些信息
    if RANK in [-1, 0]:
        # 训练停止 向控制台输出信息
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 可视化训练结果: results1.png   confusion_matrix.png 以及('F1', 'PR', 'P', 'R')曲线变化  日志信息
        for f in last, best:
            if f.exists():
                # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    # 把最好的模型在验证集上跑一边 并绘图
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=is_coco,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)  # val best model with plots
                    if is_coco:# 如果是coco数据集
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        # 记录训练终止时的日志
        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    # 释放显存
    torch.cuda.empty_cache()
    return results
```

这段函数主要打印信息并释放显存

首先当训练停止的时候回向控制台输出信息(打印训练时间、plots可视化训练结果results1.png、confusion\_matrix.png 以及(‘F1’, ‘PR’, ‘P’, ‘R’)曲线变化 、日志信息)

然后把最佳的模型取出，用这个最佳的模型跑一边验证集。再将结果保存下来，至此训练完成。若使用了超参数进化，还会进行多次训练，来完成超参数的调整

接着进行coco评价，也就是说只在coco数据集才会运行

最后释放显存

return results~

## 🚀五、执行run（）函数 

```java
'''===============================五、run（）函数=========================================='''
def run(**kwargs):
    # 执行这个脚本/ 调用train函数 / 开启训练
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        # setattr() 赋值属性，属性不存在则创建一个赋值
        setattr(opt, k, v)
    main(opt)
```

这段代码主要是使得支持指令执行这个脚本。

大家也可以看出来哈，run()函数内的内容与主函数差不多呢，都是调用了parse\_opt()函数与main()函数（其实写到这时我以为我出现了幻觉），我去查了一下，run()函数是为导入时提供的，别的模块导入了train模块，即可通过调用run()函数执行训练过程。

## 🚀六、train.py代码全部注释 

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset
在数据集上训练 yolo v5 模型
Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    训练数据为coco128 coco128数据集中有128张图片 80个类别，是规模较小的数据集
"""

'''===============================================一、导入包==================================================='''
'''======================1.导入安装好的python库====================='''
import argparse  # 解析命令行参数模块
import math  # 数学公式模块
import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time   # 时间模块 更底层
from copy import deepcopy  # 深度拷贝模块
from datetime import datetime  # datetime模块能以更方便的格式显示日期或对日期进行运算。
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np  # numpy数组操作模块
import torch # 引入torch
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import yaml  # yaml是一种直观的能够被电脑识别的的数据序列化格式，容易被人类阅读，并且容易和脚本语言交互。一般用于存储配置文件。
from torch.cuda import amp  # PyTorch amp自动混合精度训练模块
from torch.nn.parallel import DistributedDataParallel as DDP  # 多卡训练模块
from torch.optim import SGD, Adam, lr_scheduler   # tensorboard模块
from tqdm import tqdm  # 进度条模块

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()  # __file__指的是当前文件(即train.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/train.py
ROOT = FILE.parents[0]  # YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径

'''===================3..加载自定义模块============================'''
import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first


'''================4. 分布式训练初始化==========================='''
# https://pytorch.org/docs/stable/elastic/run.html该网址有详细介绍
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # -本地序号。这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))  # -进程序号。这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总共有几个 Worker
'''
   查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，
   若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配。
'''

'''===============================================二、train（）函数：训练过程==================================================='''

''' =====================1.载入参数和初始化配置信息==========================  '''
def train(hyp,  # 超参数 可以是超参数配置文件的路径或超参数字典 path/to/hyp.yaml or hyp
          opt,  # main中opt参数
          device,  # 当前设备
          callbacks  # 用于存储Loggers日志记录器中的函数，方便在每个训练阶段控制日志的记录情况
          ):
    # 从opt获取参数。日志保存路径，轮次、批次、权重、进程序号(主要用于分布式训练)等
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    '''
    1.1创建目录，设置模型、txt等保存的路径
    '''
    # Directories 获取记录训练日志的保存路径
    # 设置保存权重路径 如runs/train/exp1/weights
    w = save_dir / 'weights'  # weights dir
    # 新建文件夹 weights train evolve
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 保存训练结果的目录，如last.pt和best.pt
    last, best = w / 'last.pt', w / 'best.pt'

    '''
    1.2 读取hyp(超参数)配置文件
    '''
    # Hyperparameters 加载超参数
    if isinstance(hyp, str): # isinstance()是否是已知类型。 判断hyp是字典还是字符串
        # 若hyp是字符串，即认定为路径，则加载超参数为字典
        with open(hyp, errors='ignore') as f:
            # 加载yaml文件
            hyp = yaml.safe_load(f)  # load hyps dict 加载超参信息
    # 打印超参数 彩色字体
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    '''
    1.3 将本次运行的超参数(hyp),和选项操作(opt)给保存成yaml格式,
       保存在了每次训练得到的exp文件中，这两个yaml显示了我们本次训练所选择的超参数和opt参数，opt参数是train代码下面那一堆参数选择
    '''
    # Save run settings 保存训练中的参数hyp和opt
    with open(save_dir / 'hyp.yaml', 'w') as f:
        # 保存超参数为yaml配置文件
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        # 保存命令行参数为yaml配置文件
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        # 定义数据集字典
    data_dict = None

    '''
    1.4 加载相关日志功能:如tensorboard,logger,wandb
    '''
    # Loggers 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    if RANK in [-1, 0]:  # 如果进程编号为-1或0
        # 初始化日志记录器实例
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        # W&B # wandb为可视化参数工具
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            # 如果使用中断训练 再读取一次参数
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            # 将日志记录器中的方法与字符串进行绑定
            callbacks.register_action(k, callback=getattr(loggers, k))
    '''
    1.5 配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Config 画图
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    # 设置随机种子
    init_seeds(1 + RANK)
    # 加载数据配置信息
    with torch_distributed_zero_first(LOCAL_RANK): # torch_distributed_zero_first 同步所有进程
        data_dict = data_dict or check_dataset(data)  # check if None  check_dataset 检查数据集，如果没找到数据集则下载数据集(仅适用于项目中自带的yaml文件数据集)
    # 获取训练集、测试集图片路径
    train_path, val_path = data_dict['train'], data_dict['val']
    # nc：数据集有多少种类别
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # names: 数据集所有类别的名字，如果设置了single_cls则为一类
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # 判断类别长度和文件是否对应
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    ''' =====================2.model：加载网络模型==========================  '''
    # Model 载入模型
    # 检查文件后缀是否是.pt
    check_suffix(weights, '.pt')  # check weights
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，可以根据自己的模型选择预训练权重
    pretrained = weights.endswith('.pt')

    '''
    2.1预训练模型加载 
    '''
    if pretrained:
        # 使用预训练的话：
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果本地不存在就从google云盘中自动下载模型
            # 通常会下载失败，建议提前下载下来放进weights目录
            weights = attempt_download(weights)  # download if not found locally
        # ============加载模型以及参数================= #
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        """
        两种加载模型的方式: opt.cfg / ckpt['model'].yaml
        这两种方式的区别：区别在于是否是使用resume
        如果使用resume-断点训练: 
        将opt.cfg设为空，选择ckpt['model']yaml创建模型, 且不加载anchor。
        这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        原因：
        使用断点训练时,保存的模型会保存anchor,所以不需要加载，
        主要是预训练权重里面保存了默认coco数据集对应的anchor，
        如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor。
        """
        # ***加载模型*** #
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

        # ***以下三行是获得anchor*** #
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect

        # ***模型创建*** #
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为ture 则会少加载两个键对（anchors, anchor_grid）
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # #直接加载模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    '''
    2.2设置模型输入
    '''
    # Freeze 冻结训练的网络层
    """
    冻结模型层,设置冻结层名字即可
    作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    freeze为命令行参数，默认为0，表示不冻结
    """
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    # 首先遍历所有层
    for k, v in model.named_parameters():
        # 为所有层的参数设置梯度
        v.requires_grad = True  # train all layers
        # 判断是否需要冻结
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            # 冻结训练的层梯度不更新
            v.requires_grad = False

    # Image size 设置训练和测试图片尺寸
    # 获取模型总步长和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 检查输入图片分辨率是否能被32整除
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size 设置一次训练所选取的样本数
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
       # 确保batch size满足要求
        batch_size = check_train_batch_size(model, imgsz)

    '''
    2.3 优化器设置
    '''
    # Optimizer 优化器
    nbs = 64  # nominal batch size
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # 打印缩放后的权重衰减超参数
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 将模型分成三组（BN层的weight，卷积层的weights，biases）进行优化
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    # 遍历网络中的所有层，每遍历完一层向更深的层遍历
    for v in model.modules():
        # hasattr: 测试指定的对象是否具有给定的属性，返回一个布尔值
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            # 将层的bias添加至g2
            g2.append(v.bias)
        # YOLO v5的模型架构中只有卷积层和BN层
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            # 将BN层的权重添加至g0 未经过权重衰减
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            # 将层的weight添加至g1 经过了权重衰减
            # 这里指的是卷积层的weight
            g1.append(v.weight)

    # 选用优化器，并设置g0(bn参数)组的优化方式
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 将卷积层的参数添加至优化器 并做权重衰减
    # add_param_group()函数为添加一个参数组，同一个优化器可以更新很多个参数组，不同的参数组可以设置不同的超参数
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # 将所有的bias添加至优化器
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # 打印优化信息
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    # 在内存中删除g0 g1 g2 目的是节省空间
    del g0, g1, g2

    '''
    2.4 学习率设置
    '''
    # Scheduler  设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    if opt.linear_lr:
        # 使用线性学习率
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 使用余弦退火学习率
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # 可视化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    '''
    2.5 训练前最后准备
    '''
    # EMA 设置ema（指数移动平均），考虑历史值对参数的影响，目的是为了收敛的曲线更加平滑
    ema = ModelEMA(model) if RANK in [-1, 0] else None # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建

    # Resume 断点续训
    # 断点续训其实就是把上次训练结束的模型作为预训练模型，并从中加载参数
    start_epoch, best_fitness = 0, 0.0
    if pretrained:# 如果有预训练
        # Optimizer 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            # 将预训练模型中的参数加载进优化器
            optimizer.load_state_dict(ckpt['optimizer'])
            # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
            # 获取预训练模型中的最佳fitness，保存为best.pt
            best_fitness = ckpt['best_fitness']

        # EMA
        # 加载ema模型和updates参数,保持ema的平滑性,现在yolov5是ema和model都保存了
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs 加载训练的迭代次数
        start_epoch = ckpt['epoch'] + 1 # 从上次的epoch接着训练
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        # 如果训练的轮数小于开始的轮数
        if epochs < start_epoch:
            # 打印日志恢复训练
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            # 计算新的轮数
            epochs += ckpt['epoch']  # finetune additional epochs
        # 将预训练的相关参数从内存中删除
        del ckpt, csd

    # DP mode 使用单机多卡模式训练，目前一般不使用
    # rank为进程编号。如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  多卡归一化
    if opt.sync_bn and cuda and RANK != -1:# 多卡训练，把不同卡的数据做个同步
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    ''' =====================3.加载训练数据集==========================  '''
    '''
    3.1 创建数据集
    '''
    # Trainloader 创建训练集
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '), shuffle=True)
    '''
    返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    # 标签编号最大值
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    # 类别总数
    nb = len(train_loader)  # number of batches
    # 如果小于类别数则表示有问题
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0 验证集数据集加载
    if RANK in [-1, 0]:# 加载验证集数据加载器
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:# 没有使用resume
            # 统计dataset的label信息
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:# plots画出标签信息
                plot_labels(labels, names, save_dir)
            '''
             3.2 计算anchor
             '''
            # Anchors 计算默认锚框anchor与数据集标签框的高宽比
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                '''
                参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
            # 半进度
            model.half().float()  # pre-reduce anchor precision
        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.run('on_pretrain_routine_end')

    # DDP mode 如果rank不等于-1,则使用DistributedDataParallel模式
    if cuda and RANK != -1:
        # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    ''' =====================4.训练==========================  '''

    '''
    4.1 初始化训练需要的模型参数
    '''
    # Model attributes  根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    # box为预测框的损失
    hyp['box'] *= 3 / nl  # scale to layers
    # cls为分类的损失
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # obj为置信度损失
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # 标签平滑
    hyp['label_smoothing'] = opt.label_smoothing
    # 设置模型的类别，然后将检测的类别个数保存到模型
    model.nc = nc  # attach number of classes to model
    # 设置模型的超参数，然后将超参数保存到模型
    model.hyp = hyp  # attach hyperparameters to model
    # 从训练的样本标签得到类别权重，然后将类别权重保存至模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字，然后将分类标签保存至模型
    model.names = names

    '''
    4.2 训练热身部分
    '''
    # Start training
    t0 = time.time() # 获取当前时间
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化 map和result
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练    GradScaler + autocast
    scaler = amp.GradScaler(enabled=cuda)
    # 早停止，不更新结束训练
    stopper = EarlyStopping(patience=opt.patience)
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    # 打印日志输出信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n' # 打印训练和测试输入图片分辨率
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n' # 加载图片时调用的cpu进程数
                f"Logging results to {colorstr('bold', save_dir)}\n" # 日志目录
                f'Starting training for {epochs} epochs...') # 从哪个epoch开始训练

    '''
    4.3 开始训练
    '''
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        '''
        告诉模型现在是训练阶段 因为BN层、DropOut层、两阶段目标检测模型等
        训练阶段阶段和预测阶段进行的运算是不同的，所以要将二者分开
        model.eval()指的是预测推断阶段
        '''
        model.train()

        # Update image weights (optional, single-GPU only)  更新图片的权重
        if opt.image_weights: # 获取图片采样的权重
            # 经过一轮训练，若哪一类的不精确度高，那么这个类就会被分配一个较高的权重，来增加它被采样的概率
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 将计算出的权重换算到图片的维度，将类别的权重换算为图片的权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # 通过random.choices生成图片索引indices从而进行采样，这时图像会包含一些难识别的样本
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        # 分布式训练的设置
        # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        # 将训练数据迭代器做枚举，可以遍历出索引值
        pbar = enumerate(train_loader)
        # 训练参数的表头
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        if RANK in [-1, 0]:
            # 通过tqdm创建进度条，方便训练信息的展示
            pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        # 将优化器中的所有参数梯度设为0
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图片加载至设备 并做归一化
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup 热身训练
            '''
            热身训练(前nw次迭代),热身训练迭代的次数iteration范围[1:nw] 
            在前nw次迭代中, 根据以下方式选取accumulate和学习率
            '''
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 遍历优化器中的所有参数组
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale: # 随机改变图片的尺寸
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 前向传播
            with amp.autocast(enabled=cuda):
                # 将图片送入网络得到一个预测结果
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练,平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward 反向传播 scale为使用自动混合精度运算
            scaler.scale(loss).backward()

            # Optimize 模型会对多批数据进行累积，只有达到累计次数的时候才会更新参数，再还没有达到累积次数时 loss会不断的叠加 不会被新的反传替代
            if ni - last_opt_step >= accumulate:
                '''
                 scaler.step()首先把梯度的值unscale回来，
                 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                 否则，忽略step调用，从而保证权重不更新（不被破坏）
                '''
                scaler.step(optimizer)  # optimizer.step 参数更新
                # 更新参数
                scaler.update()
                # 完成一次累积后，再将梯度清零，方便下一次清零
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                # 计数
                last_opt_step = ni

            # Log 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in [-1, 0]:
                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # 计算显存
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # 进度条显示以上信息
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # 调用Loggers中的on_train_batch_end方法，将日志记录并生成一些记录的图片
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # 根据前面设置的学习率更新策略更新学习率
        scheduler.step()

        '''
        4.4 训练完成保存模型  
        '''

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                """
                测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                       results: [1] Precision 所有类别的平均precision(最大f1时)
                                [1] Recall 所有类别的平均recall
                                [1] map@0.5 所有类别的平均mAP@0.5
                                [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                                [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                       maps: [80] 所有类别的mAP@0.5:0.95
                """
                results, maps, _ = val.run(data_dict, # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
                                           batch_size=batch_size // WORLD_SIZE * 2, # 要保证batch_size能整除卡数
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls, # 是否是单类数据集
                                           dataloader=val_loader,
                                           save_dir=save_dir,  # 保存地址 runs/train/expn
                                           plots=False, # 是否可视化
                                           callbacks=callbacks,
                                           compute_loss=compute_loss) # 损失函数(train)

            # Update best mAP 更新best_fitness
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # 若当前的fitness大于最佳的fitness
            if fi > best_fitness:
                # 将最佳fitness更新为当前fitness
                best_fitness = fi
            # 保存验证结果
            log_vals = list(mloss) + list(results) + lr
            # 记录验证数据
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model 保存模型
            """
            保存带checkpoint的模型用于inference或resuming training
            保存模型, 还保存了epoch, results, optimizer等信息
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 将当前训练过程中的所有参数赋值给ckpt
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                # Save last, best and delete 保存每轮的模型
                torch.save(ckpt, last)
                # 如果这个模型的fitness是最佳的
                if best_fitness == fi:
                    # 保存这个最佳的模型
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                # 模型保存完毕 将变量从内存中删除
                del ckpt
                # 记录保存模型时的日志
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU 停止单卡训练
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    '''
    4.5 打印输出信息  
    '''
    # 打印一些信息
    if RANK in [-1, 0]:
        # 训练停止 向控制台输出信息
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 可视化训练结果: results1.png   confusion_matrix.png 以及('F1', 'PR', 'P', 'R')曲线变化  日志信息
        for f in last, best:
            if f.exists():
                # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    # 把最好的模型在验证集上跑一边 并绘图
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=is_coco,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)  # val best model with plots
                    if is_coco:# 如果是coco数据集
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        # 记录训练终止时的日志
        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    # 释放显存
    torch.cuda.empty_cache()
    return results


'''===============================================三、设置opt参数==================================================='''

""""
    opt参数解析：
    cfg:                               模型配置文件，网络结构
    data:                              数据集配置文件，数据集路径，类名等
    hyp:                               超参数文件
    epochs:                            训练总轮次
    batch-size:                        批次大小
    img-size:                          输入图片分辨率大小
    rect:                              是否采用矩形训练，默认False
    resume:                            接着打断训练上次的结果接着训练
    nosave:                            不保存模型，默认False
    notest:                            不进行test，默认False
    noautoanchor:                      不自动调整anchor，默认False
    evolve:                            是否进行超参数进化，默认False
    bucket:                            谷歌云盘bucket，一般不会用到
    cache-images:                      是否提前缓存图片到内存，以加快训练速度，默认False
    weights:                           加载的权重文件
    name:                              数据集名字，如果设置：results.txt to results_name.txt，默认无
    device:                            训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    multi-scale:                       是否进行多尺度训练，默认False
    single-cls:                        数据集是否只有一个类别，默认False
    adam:                              是否使用adam优化器
    sync-bn:                           是否使用跨卡同步BN,在DDP模式使用
    local_rank:                        gpu编号
    logdir:                            存放日志的目录
    workers:                           dataloader的最大worker数量
"""

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 预训练权重文件
    parser.add_argument('--weights', type=str, default=ROOT / 'pretrained/yolov5s.pt', help='initial weights path')
    # 训练模型
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    # 训练路径，包括训练集，验证集，测试集的路径，类别总数等
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire_data.yaml', help='dataset.yaml path')
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # epochs: 训练轮次， 默认轮次为300次
    parser.add_argument('--epochs', type=int, default=300)
    # batchsize: 训练批次， 默认bs=16
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    # imagesize: 设置图片大小, 默认640*640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # rect: 是否采用矩形训练，默认为False
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # resume: 是否接着上次的训练结果，继续训练
    # 矩形训练：将比例相近的图片放在一个batch（由于batch里面的图片shape是一样的）
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # nosave: 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # best.pt/ last.pt 不建议运行代码添加 --nosave
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # noval: 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # noautoanchor: 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # evolve: 参数进化， 遗传算法调参
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # bucket: 谷歌优盘 / 一般用不到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # cache: 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # mage-weights: 使用图片采样策略，默认不使用
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # device: 设备选择
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # multi-scale 是否进行多尺度训练
    parser.add_argument('--multi-scale', default=True, help='vary img-size +/- 50%%')
    # single-cls: 数据集是否多类/默认True
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # optimizer: 优化器选择 / 提供了三种优化器
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # sync-bn: 是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # dataloader的最大worker数量 （使用多线程加载图片）
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    # 训练结果的保存路径
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # 训练结果的文件名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 项目位置是否存在 / 默认是都不存在
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 四元数据加载器: 允许在较低 --img 尺寸下进行更高 --img 尺寸训练的一些好处。
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # cos-lr: 余弦学习率
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # 早停止耐心次数 / 100次不更新就停止训练
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # --freeze冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    # --save-period 多少个epoch保存一下checkpoint
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # --local_rank 进程编号 / 多卡使用
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # Weights & Biases arguments
    # 在线可视化工具，类似于tensorboard工具
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    # 使用数据的版本
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

'''===============================================四、main（）函数==================================================='''
def main(opt, callbacks=Callbacks()):
    '''
    4.1  检查分布式训练环境
    '''
    # Checks
    if RANK in [-1, 0]:  # 若进程编号为-1或0
        # 输出所有训练参数 / 参数以彩色的方式表现
        print_args(FILE.stem, opt)
        # 检测YOLO v5的github仓库是否更新，若已更新，给出提示
        check_git_status()
        # 检查requirements.txt所需包是否都满足
        check_requirements(exclude=['thop'])
    '''
    4.2  判断是否断点续训
    '''
    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        # isinstance()是否是已经知道的类型
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        # 判断是否为文件，若不是文件抛出异常
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt.yaml是训练时的命令行参数文件
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            # 超参数替换，将训练时的命令行参数加载进opt参数对象中
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        # 打印从ckpt恢复断点训练信息
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        # 不使用断点续训，就从文件中读取相关参数
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        # 如果模型文件和权重文件为空，弹出警告
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 如果要进行超参数进化，重建保存路径
        if opt.evolve:
            # 设置新的项目输出目录
            opt.project = str(ROOT / 'runs/evolve')
            # 将resume传递给exist_ok
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        # 根据opt.project生成目录，并赋值给opt.save_dir  如: runs/train/exp1
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    '''
    4.3  判断是否分布式训练
    '''
    # DDP mode -->  支持多机多卡、分布式训练
    # 选择程序装载的位置
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 当进程内的GPU编号不为-1时，才会进入DDP
    if LOCAL_RANK != -1:
        #  用于DDP训练的GPU数量不足
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # WORLD_SIZE表示全局的进程数
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        # 不能使用图片采样策略
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        # 不能使用超参数进化
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        # 设置装载程序设备
        torch.cuda.set_device(LOCAL_RANK)
        # 保存装载程序的设备
        device = torch.device('cuda', LOCAL_RANK)
        # torch.distributed是用于多GPU训练的模块
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    '''
    4.4  判断是否进化训练
    '''
    # Train 训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:# 如果不使用超参数进化
        # 开始训练
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            # 如果全局进程数大于1并且RANK等于0
            # 日志输出 销毁进程组
            LOGGER.info('Destroying process group... ')
            # 训练完毕，销毁所有进程
            dist.destroy_process_group()

    # Evolve hyperparameters (optional) 遗传进化算法，边进化边训练
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数列表(突变范围 - 最小值 - 最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)
        # 加载默认超参数
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

            """
            遗传算法调参：遵循适者生存、优胜劣汰的法则，即寻优过程中保留有用的，去除无用的。
            遗传算法需要提前设置4个参数: 群体大小/进化代数/交叉概率/变异概率
            """
        # 选择超参数的遗传迭代次数 默认为迭代300次
        for _ in range(opt.evolve):  # generations to evolve
            # 如果evolve.csv文件存在
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                # 选择超参进化方式，只用single和weighted两种
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 选取至多前五次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                # fitness()为x前四项加权 [P, R, mAP@0.5, mAP@0.5:0.95]
                # np.argsort只能从小到大排序, 添加负号实现从大到小排序, 算是排序的一个代码技巧
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据(mp, mr, map50, map)的加权和来作为权重计算hyp权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # 根据权重的几率随机挑选适应度历史前5的其中一个
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    # 对hyp乘上对应的权重融合层一个hpy, 再取平均(除以权重和)
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate 突变（超参数进化）
                mp, s = 0.8, 0.2  # mutation probability, sigma：突变概率
                npr = np.random
                # 根据时间设置随机数种子
                npr.seed(int(time.time()))
                # 获取突变初始值, 也就是meta三个值的第一个数据
                # 三个数值分别对应着: 变异初始概率, 最低限值, 最大限值(mutation scale 0-1, lower_limit, upper_limit)
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                # 确保至少其中有一个超参变异了
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits 限制hyp在规定范围内
            for k, v in meta.items():
                # 这里的hyp是超参数配置文件对象
                # 而这里的k和v是在元超参数中遍历出来的
                # hyp的v是一个数，而元超参数的v是一个元组
                hyp[k] = max(hyp[k], v[1])  # 先限定最小值，选择二者之间的大值 ，这一步是为了防止hyp中的值过小
                hyp[k] = min(hyp[k], v[2])  # 再限定最大值，选择二者之间的小值
                hyp[k] = round(hyp[k], 5)  # 四舍五入到小数点后五位
                # 最后的值应该是 hyp中的值与 meta的最大值之间的较小者

            # Train mutation 使用突变后的参超，测试其效果
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            # 将结果写入results，并将对应的hyp写到evolve.txt，evolve.txt中每一行为一次进化的结果
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            # 保存hyp到yaml文件
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results 将结果可视化 / 输出保存信息
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')

'''===============================================五、run（）函数==================================================='''
def run(**kwargs):
    # 执行这个脚本/ 调用train函数 / 开启训练
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        # setattr() 赋值属性，属性不存在则创建一个赋值
        setattr(opt, k, v)
    main(opt)

# python train.py --data fire_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 2 --device cpu
# python train.py --data fire_data.yaml --cfg mask_yolov5l.yaml --weights pretrained/yolov5l.pt --epoch 100 --batch-size 2
# python train.py --data fire_data.yaml --cfg mask_yolov5m.yaml --weights pretrained/yolov5m.pt --epoch 100 --batch-size 2
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
```

本文参考：

[【YOLOV5-5.x 源码解读】train.py][YOLOV5-5.x _train.py]  
![](https://i-blog.csdnimg.cn/blog_migrate/a556bc28602a28c086d7e5e2087c2978.gif)


[mirrors _ ultralytics _ yolov5 _ GitCode]: https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator
[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764
[Link 1]: #%E5%89%8D%E8%A8%80
[Link 2]: #main-toc
[Link 3]: #%F0%9F%9A%80%E4%B8%80%E3%80%81%E5%AF%BC%E5%8C%85%E5%92%8C%E5%9F%BA%E6%9C%AC%E9%85%8D%E7%BD%AE
[1.1 Usage]: #1.1%20Usage
[1.2 _python]: #1.2%20%E5%AF%BC%E5%85%A5%E5%AE%89%E8%A3%85%E5%A5%BD%E7%9A%84python%E5%BA%93
[1.3]: #1.3%20%E8%8E%B7%E5%8F%96%E5%BD%93%E5%89%8D%E6%96%87%E4%BB%B6%E7%9A%84%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84
[1.4]: #1.4%20%E5%8A%A0%E8%BD%BD%E8%87%AA%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9D%97
[1.5]: #1.5%C2%A0%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%88%9D%E5%A7%8B%E5%8C%96
[main]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E6%89%A7%E8%A1%8Cmain%EF%BC%88%EF%BC%89%E5%87%BD%E6%95%B0
[2.1]: #2.1%C2%A0%E6%A3%80%E6%9F%A5%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E7%8E%AF%E5%A2%83
[2.2]: #2.2%20%E5%88%A4%E6%96%AD%E6%98%AF%E5%90%A6%E6%96%AD%E7%82%B9%E7%BB%AD%E8%AE%AD
[2.3]: #2.3%20%E5%88%A4%E6%96%AD%E6%98%AF%E5%90%A6%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83
[2.4]: #2.4%C2%A0%20%E5%88%A4%E6%96%AD%E6%98%AF%E5%90%A6%E8%BF%9B%E5%8C%96%E8%AE%AD%E7%BB%83
[opt]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E8%AE%BE%E7%BD%AEopt%E5%8F%82%E6%95%B0
[train]: #%F0%9F%9A%80%E5%9B%9B%E3%80%81%E6%89%A7%E8%A1%8Ctrain%EF%BC%88%EF%BC%89%E5%87%BD%E6%95%B0
[4.1]: #4.1%20%E5%8A%A0%E8%BD%BD%E5%8F%82%E6%95%B0%E5%92%8C%E5%88%9D%E5%A7%8B%E5%8C%96%E9%85%8D%E7%BD%AE%E4%BF%A1%E6%81%AF
[4.1.1]: #4.1.1%20%E8%BD%BD%E5%85%A5%E5%8F%82%E6%95%B0
[4.1.2]: #4.1.2%20%E5%88%9B%E5%BB%BA%E8%AE%AD%E7%BB%83%E6%9D%83%E9%87%8D%E7%9B%AE%E5%BD%95%E5%92%8C%E4%BF%9D%E5%AD%98%E8%B7%AF%E5%BE%84
[4.1.3]: #4.1.3%20%E8%AF%BB%E5%8F%96%E8%B6%85%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6
[4.1.4]: #4.1.4%20%E8%AE%BE%E7%BD%AE%E5%8F%82%E6%95%B0%E7%9A%84%E4%BF%9D%E5%AD%98%E8%B7%AF%E5%BE%84
[4.1.5]: #4.1.5%20%E5%8A%A0%E8%BD%BD%E6%97%A5%E5%BF%97%E4%BF%A1%E6%81%AF
[4.1.6]: #4.1.6%20%E5%8A%A0%E8%BD%BD%E5%85%B6%E5%AE%83%E5%8F%82%E6%95%B0
[4.2]: #4.2%20%E5%8A%A0%E8%BD%BD%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B
[4.2.1]: #4.2.1%20%E5%8A%A0%E8%BD%BD%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B
[4.2.2]: #4.2.2%20%E8%AE%BE%E7%BD%AE%E5%86%BB%E7%BB%93%E5%B1%82
[4.2.3]: #%C2%A04.2.3%20%E8%AE%BE%E7%BD%AE%E4%BC%98%E5%8C%96%E5%99%A8
[4.2.4]: #4.2.4%20%E8%AE%BE%E7%BD%AE%E5%AD%A6%E4%B9%A0%E7%8E%87
[4.2.5]: #4.2.5%20%E8%AE%AD%E7%BB%83%E5%89%8D%E6%9C%80%E5%90%8E%E5%87%86%E5%A4%87
[4.3]: #4.3%20%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE%E9%9B%86
[4.3.1]: #4.3.1%20%E5%88%9B%E5%BB%BA%E6%95%B0%E6%8D%AE%E9%9B%86
[4.3.2 _anchor]: #4.3.2%20%E8%AE%A1%E7%AE%97anchor
[4.4]: #4.4%20%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B
[4.4.1]: #4.4.1%20%E5%88%9D%E5%A7%8B%E5%8C%96%E8%AE%AD%E7%BB%83%E9%9C%80%E8%A6%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0
[4.4.2]: #4.4.2%20%E8%AE%AD%E7%BB%83%E7%83%AD%E8%BA%AB%E9%83%A8%E5%88%86
[4.4.3]: #4.4.3%20%E5%BC%80%E5%A7%8B%E8%AE%AD%E7%BB%83
[4.4.4]: #4.4.4%20%E8%AE%AD%E7%BB%83%E5%AE%8C%E6%88%90%E4%BF%9D%E5%AD%98%E6%A8%A1%E5%9E%8B
[4.5]: #4.5%20%E6%89%93%E5%8D%B0%E4%BF%A1%E6%81%AF%E5%B9%B6%E9%87%8A%E6%94%BE%E6%98%BE%E5%AD%98
[run]: #%F0%9F%9A%80%E4%BA%94%E3%80%81%E6%89%A7%E8%A1%8Crun%EF%BC%88%EF%BC%89%E5%87%BD%E6%95%B0
[train.py]: #%F0%9F%9A%80%E5%85%AD%E3%80%81train.py%E4%BB%A3%E7%A0%81%E5%85%A8%E9%83%A8%E6%B3%A8%E9%87%8A
[YOLOv5 _v5.0-v7.0]: https://yolov5.blog.csdn.net/article/details/124411509
[Link 4]: https://so.csdn.net/so/search?q=%E5%8F%AC%E5%9B%9E%E7%8E%87&spm=1001.2101.3001.7020
[YOLOV5-5.x _train.py]: https://blog.csdn.net/qq_38253797/article/details/119733964