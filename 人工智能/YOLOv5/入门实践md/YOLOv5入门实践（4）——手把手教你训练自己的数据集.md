#### ![](https://i-blog.csdnimg.cn/blog_migrate/d8d15792901a80e343ee53d6bd39bde8.gif) 

![](https://i-blog.csdnimg.cn/blog_migrate/18004cfb9dd8234e8d42102878d17bcb.jpeg)

## 前言 

在上一篇文章中我们介绍了如何划分数据集，划分好之后我们的前期准备工作就已经全部完成了，下面开始训练自己的数据集吧！

![](https://i-blog.csdnimg.cn/blog_migrate/ab772248d04a94e8b7fa705eb0735031.gif)

前期回顾：

[YOLOv5入门实践（1）——手把手带你环境配置搭建][YOLOv5_1]

[YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集][YOLOv5_2_labelimg]

[YOLOv5入门实践（3）——手把手教你划分自己的数据集][YOLOv5_3]

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1 1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

目录

[前言][Link 1]

[ 一、配置文件][Link 2]

[1.1 修改数据集配置文件][1.1]

[1.2 修改模型配置文件][1.2]

[二、训练模型][Link 3]

[三、性能评价指标][Link 4]

![](https://i-blog.csdnimg.cn/blog_migrate/38d4ba903d66d8adb560bc2b86395f30.gif)

## 一、配置文件 

在训练前我们首先来配置文件，通过之前的学习（[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]），我们知道YOLOv5训练数据都是通过调用yaml文件里我们已经整理好的数据。在这里，我们首先需要修改两个yaml文件中的参数。一个是data目录下的相应的yaml文件（数据集配置文件），一个是model目录文件下的相应的yaml文件（模型配置文件）。

### 1.1 修改数据集配置文件 

首先在data的目录下新建一个yaml文件，自定义命名（嫌麻烦的话可以直接复制voc.yaml文件，重命名然后在文件内直接修改。）。

![](https://i-blog.csdnimg.cn/blog_migrate/5f34f8d74f5aa1c22584f2071528da5e.png)

然后修改文件内的路径和参数。

train和val就是上一篇文章中通过split划分好的数据集文件（最好要填绝对路径，有时候由目录结构的问题会莫名奇妙的报错）

下面是两个参数，根据自己数据集的检测目标个数和名字来设定：

 *  nc: 存放检测目标类别个数
 *  name： 存放检测目标类别的名字（个数和nc对应）

这里我做的是火焰检测，所以目标类别只有一个。

![](https://i-blog.csdnimg.cn/blog_migrate/9165bc4e66476abf6ec9a570b4eff567.png)

> 注意：也可以在data目录下的coco.yaml上修改自己的路径、类别数和类别名称。
> 
> 若在训练时报错，解决方法是：冒号后面需要加空格，否则会被认为是字符串而不是字典。

### 1.2 修改模型配置文件 

在model文件夹下有4种不同大小的网络模型，分别是YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x，这几个模型的结构基本一样，不同的是depth\_multiple模型深度和width\_multiple模型宽度这两个参数。

我们本次使用的是yolov5s.pt这个预训练权重，同上修改data目录下的yaml文件一样，我们最好将yolov5s.yaml文件复制一份，然后将其重命名，我将其重命名为yolov5\_fire.yaml。

![](https://i-blog.csdnimg.cn/blog_migrate/5ef3383c63ae130a4869b408f36de5fd.png)

同样，这里改一下nc

![](https://i-blog.csdnimg.cn/blog_migrate/df0d617962ef01fd1f738d8b9c5b8c9c.png)

## 二、训练模型 

训练模型是通过train.py文件，在训练前我们先介绍一下文件内的参数

```java
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_fire.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```

> opt参数解析：
> 
>  *  cfg: 模型配置文件，网络结构
>  *  data: 数据集配置文件，数据集路径，类名等
>  *  hyp:  超参数文件
>  *  epochs:  训练总轮次
>  *  batch-size: 批次大小
>  *  img-size:  输入图片分辨率大小
>  *  rect:  是否采用矩形训练，默认False
>  *  resume:  接着打断训练上次的结果接着训练
>  *  nosave: 不保存模型，默认False
>  *  notest:  不进行test，默认False
>  *  noautoanchor:  不自动调整anchor，默认False
>  *  evolve: 是否进行超参数进化，默认False
>  *  bucket: 谷歌云盘bucket，一般不会用到
>  *  cache-images: 是否提前缓存图片到内存，以加快训练速度，默认False
>  *  weights:  加载的权重文件
>  *  name: 数据集名字，如果设置：results.txt to results\_name.txt，默认无
>  *  device:  训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
>  *  multi-scale: 是否进行多尺度训练，默认False
>  *  single-cls:  数据集是否只有一个类别，默认False
>  *  adam: 是否使用adam优化器
>  *  sync-bn:  是否使用跨卡同步BN,在DDP模式使用
>  *  local\_rank:  gpu编号
>  *  logdir: 存放日志的目录
>  *  workers: dataloader的最大worker数量
> 
> （train.py更多学习请看：[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]）

参数虽然很多，但是需要我们修改的很少：

![](https://i-blog.csdnimg.cn/blog_migrate/42ecba8ecb8cb4dfd10aa356b79be5f2.png)

```java
parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s_fire.pt', help='initial weights path')
```

 --weight ：先选用官方的yolov5s.pt权重，当自己的训练完成后可更换为自己的权重

```java
parser.add_argument('--cfg', type=str, default='models/yolov5s_fire.yaml', help='model.yaml path')
```

 --cfg：选用上一步model目录下我们刚才改好的模型配置文件

```java
parser.add_argument('--data', type=str, default=ROOT / 'data/fire.yaml', help='dataset.yaml path')
```

 --data：选用上一步data目录下我们刚才改好的数据集配置文件

```java
parser.add_argument('--epochs', type=int, default=300)
```

 --epoch：指的就是训练过程中整个数据集将被迭代多少轮，默认是300，显卡不行就调小点

```java
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
```

--batch-size：一次看完多少张图片才进行权重更新，默认是16，显卡不行就调小点

```java
parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
```

--workers:dataloader的最大worker数量，一般用来处理多线程问题，默认是8，显卡不行就调小点

以上都设置好了就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/398a85afbbe70a20e1aa0ea0726e5064.png)

若干个hours之后~

训练结果会保存在runs的train文件里。

![](https://i-blog.csdnimg.cn/blog_migrate/3deb47751c305d5d7bd8cb92bd207253.png)

至此，我们的训练就全部完成了~

## 三、性能评价指标 

模型性能评价好坏主要看训练完成后得到的results图（先忽略我的精度），我们看看这里面都有啥：

![](https://i-blog.csdnimg.cn/blog_migrate/556cdf395f86a5afd864de614240801e.png)

 *  box\_loss： 推测为GIoU损失函数均值，越小方框越准；
 *  obj\_loss： 推测为目标检测loss均值，越小目标检测越准；
 *  cls\_loss： 推测为分类loss均值，越小分类越准；
 *  precision：准确率（找对的/找到的）；
 *  recall：召回率（找对的/该找对的）；
 *  mAP@0.5 & mAP@0.5:0.95： 就是mAP是用Precision和Recall作为两轴作图后围成的面积，m表示平均，@后面的数表示判定iou为正负样本的阈值，@0.5:0.95表示阈值取0.5:0.05:0.95后取均值。（0.5是iou阈值=0.5时mAP的值），mAP只是一个形容PR曲线面积的代替词叫做平均准确率，越高越好。

> 本文参考：
> 
> [ yolov5训练相关参数解释][yolov5]
> 
> [目标检测---教你利用yolov5训练自己的目标检测模型\_][---_yolov5]

![](https://i-blog.csdnimg.cn/blog_migrate/163f360ea8e8c139a9dc4e5df38b4063.gif)


[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501
[YOLOv5_2_labelimg]: https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501
[YOLOv5_3]: https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501
[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1 1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501
[Link 1]: #%E5%89%8D%E8%A8%80
[Link 2]: #%C2%A0%E4%B8%80%E3%80%81%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6
[1.1]: #1.1%20%E4%BF%AE%E6%94%B9%E6%95%B0%E6%8D%AE%E9%9B%86%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6
[1.2]: #1.2%20%E4%BF%AE%E6%94%B9%E6%A8%A1%E5%9E%8B%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6
[Link 3]: #%E4%BA%8C%E3%80%81%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B
[Link 4]: #%C2%A0%E4%B8%89%E3%80%81%E6%80%A7%E8%83%BD%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87
[yolov5]: https://blog.csdn.net/weixin_41990671/article/details/107300314?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov5results%E5%9B%BE%E7%89%87%E5%8F%82%E6%95%B0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-107300314.142%5Ev82%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&spm=1018.2226.3001.4187
[---_yolov5]: https://blog.csdn.net/didiaopao/article/details/119954291?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168100760816800197099112%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168100760816800197099112&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-119954291-null-null.142%5Ev82%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E8%AE%AD%E7%BB%83%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4187