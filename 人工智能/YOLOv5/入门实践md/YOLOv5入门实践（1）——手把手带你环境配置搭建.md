#### ![](https://i-blog.csdnimg.cn/blog_migrate/bfe7f1e515b26048ba27acb760d78440.gif) 

## ![](https://i-blog.csdnimg.cn/blog_migrate/15b4b8bb2e8eec5fdc7cdc2bc8f15dc3.jpeg) 

## 前言 

这两天我将pycharm社区版换成了专业版，也顺带着把环境从CPU改成了GPU版，本篇文章也就是我个人配置过程的一个简单记录，希望能够帮到大家啦~

![](https://i-blog.csdnimg.cn/blog_migrate/5b56462435e930a94f45ebcaac033865.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

## 目录 

[前言][Link 1]

[一、了解所需配置][Link 2]

[1.1 CUDA][]

[1.2 cuDNN][]

[1.3 Anconda][]

[1.4 pycharm][]

[1.5 pytorch][]

[二、安装CUDA 和cuDNN][CUDA _cuDNN]

[2.1 CUDA的下载与安装][2.1 CUDA]

[2.2 cuDNN的下载与安装][2.2 cuDNN]

[三、安装Anaconda ][Anaconda]

[四、安装pytorch ][pytorch]

[五、配置YOLOv5环境][YOLOv5 1]

[六、测试][Link 3]

![](https://i-blog.csdnimg.cn/blog_migrate/d81340937af65638e899750006ba4c41.gif)

## 一、了解所需配置 

### 1.1 CUDA 

2006年，NVIDIA公司发布了CUDA(Compute Unified Device Architecture)，是一种新的操作GPU计算的硬件和软件架构，是建立在NVIDIA的GPUs上的一个通用并行计算平台和编程模型，它提供了GPU编程的简易接口，基于CUDA编程可以构建基于GPU计算的应用程序，利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。它将GPU视作一个数据并行计算设备，而且无需把这些计算映射到图形API。操作系统的多任务机制可以同时管理CUDA访问GPU和图形程序的运行库，其计算特性支持利用CUDA直观地编写GPU核心程序。

### 1.2 cuDNN 

cuDNN是NVIDIACUDA®深度神经网络库，是GPU加速的用于深度神经网络的原语库。cuDNN为标准例程提供了高度优化的实现，例如向前和向后卷积，池化，规范化和激活层。  
全球的深度学习研究人员和框架开发人员都依赖cuDNN来实现高性能GPU加速。它使他们可以专注于训练神经网络和开发软件应用程序，而不必花时间在底层GPU性能调整上。cuDNN的加快广泛使用的深度学习框架，包括Caffe2，Chainer，Keras，MATLAB，MxNet，PyTorch和TensorFlow。已将cuDNN集成到框架中的NVIDIA优化深度学习框架容器，访问NVIDIA GPU CLOUD了解更多信息并开始使用。

### 1.3 Anconda 

Anaconda指的是一个开源的Python发行版本，其包含了conda、Python等180多个科学包及其依赖项。 因为包含了大量的科学包，Anaconda 的下载文件比较大（约 531 MB），如果只需要某些包，或者需要节省带宽或存储空间，也可以使用Miniconda这个较小的发行版（仅包含conda和 Python）。  
Anaconda包括Conda、Python以及一大堆安装好的工具包，比如：numpy、pandas等

### 1.4 pycharm 

pycharm是一个用于计算机编程的集成开发环境，主要用于python语言开发，并支持使用Django进行网页开发。简单来说就是人工智能的便捷语言。

### 1.5 pytorch 

[PyTorch][]是一个开源的Python机器学习库，其前身是2002年诞生于纽约大学 的Torch。它是美国Facebook公司使用python语言开发的一个深度学习的框架，2017年1月，Facebook人工智能研究院（FAIR）在GitHub上开源了PyTorch。

想进一步学习pytorch的友友，欢迎关注我的专栏噢！→[Pytorch\_路人贾'ω'的博客-CSDN博客][Pytorch_-CSDN]

## 二、安装CUDA 和cuDNN 

官方教程

CUDA：[https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html][https_docs.nvidia.com_cuda_cuda-installation-guide-microsoft-windows_index.html]

cuDNN：[Installation Guide :: NVIDIA Deep Learning cuDNN Documentation][Installation Guide _ NVIDIA Deep Learning cuDNN Documentation]

### 2.1 CUDA的下载与安装 

1.首先查看自己CUDA的版本，有以下两种方法：

（1）打开[nvidia][]（桌面右键）->选择左下角的系统信息->组件

![](https://i-blog.csdnimg.cn/blog_migrate/1050bf13749fd5bf92b0d8ae377ed09e.png)

![](https://i-blog.csdnimg.cn/blog_migrate/6137d9488ab9ad0d198d49da574da6cb.png)

（2）直接在cmd中输入

```java
nvidia-smi
```

这里就可以直接查看啦

![](https://i-blog.csdnimg.cn/blog_migrate/2a302a9b381caf9594d7557c1f995e73.png)

 2.然后开始进入官网下载对应版本，下载地址→ [官方驱动 | NVIDIA][_ NVIDIA]

![](https://i-blog.csdnimg.cn/blog_migrate/7391e8972d3abf9d961ed359e4dc81b0.png)

根据自己查到的版本下载对应既可。

然后就是漫长的等待ing

![](https://i-blog.csdnimg.cn/blog_migrate/227fbbcc762cefc742d474cdbaf071f1.png)

 3.下载完了就开始安装

![](https://i-blog.csdnimg.cn/blog_migrate/10d1695bc7761dd1c046996300160f30.png)

点击下一步

![](https://i-blog.csdnimg.cn/blog_migrate/3eaaf03b0c2ef064907ca2a4807adffe.png)

这两个可以不用勾选

![](https://i-blog.csdnimg.cn/blog_migrate/bc411ea9d6e69885fb6070dd04ef359e.png)

4.查看环境变量

点击开始-->搜索高级系统设置-->查看环境变量

【如果没有需要自己添加】

![](https://i-blog.csdnimg.cn/blog_migrate/5a1e473d8503ef5c6f2a4def97ddd410.png)

![](https://i-blog.csdnimg.cn/blog_migrate/8cdcd56a22d97016a624bd4b9845402d.png)

![](https://i-blog.csdnimg.cn/blog_migrate/ebd25025e1c5e75841d52313ec2807df.png)

一共四个系统变量，都是自动生成的，但是有时后两个系统变量可能不会自动生成，需要自己添加上去，添加时注意路径。

5.验证CUDA是否安装成功

win+r，运行cmd，输入

```java
nvcc --version 
OR
nvcc -V
```

即可查看版本号

![](https://i-blog.csdnimg.cn/blog_migrate/0291e43c0565e0af249689535014e343.png) 输入

```java
set cuda
```

即可查看 CUDA 设置的环境变量 ![](https://i-blog.csdnimg.cn/blog_migrate/423decdb23349405bcb4d0ef6974884d.png)

至此，CUDA 就已安装完成。但是在完成张量加速运算时还需要cuDNN的辅助，所以接下来我们来安装cuDNN。

### 2.2 cuDNN的下载与安装 

1.查看与CUDA对应的cuDNN版本

![](https://i-blog.csdnimg.cn/blog_migrate/a10b8951b983e4a84b31bbd8c1f4f120.png)

2.在官网上下载。官网地址→[cuDNN Download | NVIDIA Developer][cuDNN Download _ NVIDIA Developer]

点击注册

![](https://i-blog.csdnimg.cn/blog_migrate/236ca406c3b02a24d87f1001bf8f43f4.png) 注册成功

![](https://i-blog.csdnimg.cn/blog_migrate/05a5bbf376b9e27314c3cc9265e2c4b3.png)

3.开始下载

![](https://i-blog.csdnimg.cn/blog_migrate/4e39c25d1ede2be673ec8ae17c94c8b3.png)

 4.解压文件

![](https://i-blog.csdnimg.cn/blog_migrate/92abb7622d4486cb48e1d5b95f624df0.png)

我们下载后发现其实cudnn不是一个exe文件，而是一个压缩包，解压后，有三个文件夹，把三个文件夹拷贝到cuda的安装目录下。

![](https://i-blog.csdnimg.cn/blog_migrate/3477cd89aa0baa116814958758e92014.png) cuDNN 其实是 CUDA 的一个补丁，专为深度学习运算进行优化的。然后再添加环境变量

5.添加至系统变量

往系统环境变量中的 path 添加如下路径（根据自己的路径进行修改） ![](https://i-blog.csdnimg.cn/blog_migrate/b8cbfa5985e371cb08ceaf0e032c51f0.png)

7.验证cuDNN是否安装成功

win+r，启动cmd，cd到安装目录下的.\\extras\\demo\_suite，输入

```java
原目录.\extras\demo_suite
```

然后分别输入.\\bandwidthTest.exe和.\\deviceQuery.exe（进到目录后需要直接输“bandwidthTest.exe”和“deviceQuery.exe”）

```java
.\bandwidthTest.exe
```

```java
.\deviceQuery.exe
```

得到下图：

![](https://i-blog.csdnimg.cn/blog_migrate/8d80eb272d636c7a83b27cbc1b191718.png)

![](https://i-blog.csdnimg.cn/blog_migrate/01719853b63a55b09a1e15acd93a381d.png)

至此，CUDA和cuDNN已全部安装完毕~

## 三、安装Anaconda  

因为我的电脑已经有了Anaconda ，所以没有再安装。没有安装的可以看看这个教程：

[最新Anaconda3的安装配置及使用教程（详细过程）\_HowieXue的博客-CSDN博客][Anaconda3_HowieXue_-CSDN]

## 四、安装pytorch 

同样，pytorch我电脑上也安装过了（不然咋出的专栏呢（手动狗头））。没有安装的推荐大家看我同门的这篇文章，步骤非常详细：[Win11上Pytorch的安装并在Pycharm上调用PyTorch最新超详细\_win11安装pytorch][Win11_Pytorch_Pycharm_PyTorch_win11_pytorch]

## 五、配置YOLOv5环境 

1.yolov5的源码下载

下载地址：[mirrors / ultralytics / yolov5 · GitCode][mirrors _ ultralytics _ yolov5 _ GitCode]

方法一：git clone到本地本地仓库  
\[指令\]：`git clone https://github.com/ultralytics/yolov5`

方法二：直接安装压缩包  
没有安装 git 的话，可以直接点击“克隆”下载压缩包

![](https://i-blog.csdnimg.cn/blog_migrate/033151eef87908a715e79f1af32653e7.png)

2.预训练模型下载

为了缩短网络的训练时间，并达到更好的精度，我们一般加载预训练权重进行网络的训练。  
YOLOv5给我们提供了几个预训练权重，我们可以对应我们不同的需求选择不同的版本的预训练权重。在实际场景中是比较看这种速度，所以YOLOv5s是比较常用的。

将安装好的预训练模型放在YOLO文件下。

![](https://i-blog.csdnimg.cn/blog_migrate/791c24451592df13bc612a9ef7a583a1.png)

  
3.安装yolov5的依赖项 

首先创建虚拟环境并激活。conda常用指令如下：

 *  创建虚拟环境：
    
    ```java
    conda create -n [虚拟环境名] python=[版本]
    ```

![](https://i-blog.csdnimg.cn/blog_migrate/a6fb898a61e7d2131468db48c8adf32c.png) 点“y”

![](https://i-blog.csdnimg.cn/blog_migrate/81c2bff566b033af49175a8adfe9d8dd.png)

 *  显示虚拟环境：

```java
conda env list
```

![](https://i-blog.csdnimg.cn/blog_migrate/dce403a0cc19fdb9069d314a218db6f6.png)

 *  激活虚拟环境：

```java
conda activate + [虚拟环境名]
```

![](https://i-blog.csdnimg.cn/blog_migrate/e4bab170a72242c64b6a5950e4338277.png)

4.安装pytorch-gup版的环境

由于pytorch的官网在国外，下载相关的环境包是比较慢的，所以我们给环境换源。在pytorch环境下执行如下的命名给环境换清华源。

```java
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

这里要注意网速一定要好，不然就总报错。下图就是本人血泪史。

![](https://i-blog.csdnimg.cn/blog_migrate/8ece52d56c885f75440e262174c17c6b.png)

然后就安装完啦

![](https://i-blog.csdnimg.cn/blog_migrate/613ad6d2b2055cf7986a89e00ce05cc0.png)

打开我们下载好的源码，点击设置setting

![](https://i-blog.csdnimg.cn/blog_migrate/bd03b17bd7697b11d962c5c131a142e3.png)

按以下步骤就OK啦！

![](https://i-blog.csdnimg.cn/blog_migrate/f9d2f6877f859a0b97411aa31f922cf2.png)

## 六、测试 

1.我们先运行detect.py

![](https://i-blog.csdnimg.cn/blog_migrate/95cc88080f109bb83c1b1eca953556c7.png)

这时会发现出现错误：

AttributeError: 'Upsample' object has no attribute 'recompute scale\_factor'

![](https://i-blog.csdnimg.cn/blog_migrate/61d94d1a5388d9b6a3e8e658c8237498.png)

解决方法：

点进蓝色的文件里下图对应位置，更改forward函数，复制一遍，去掉下面一行的代码

![](https://i-blog.csdnimg.cn/blog_migrate/0a04300872f35b0567505a1219dfee1d.jpeg)

再点击run，结果就保存在runs的detect文件下了

![](https://i-blog.csdnimg.cn/blog_migrate/11bede5aebac87ce44abf891f681a75d.png) ![](https://i-blog.csdnimg.cn/blog_migrate/87f5aa940bb9c3ebef15f68754462280.png)

2.我们再运行train.py

![](https://i-blog.csdnimg.cn/blog_migrate/fa3626520e61f3a7a4ff1843d637b934.png)

同样会发生报错

0SError: \[winError 1455\]页面文件太小，无法完成操作。Error loading"D:\\Anaconda3\\envslyolov5-6.1lib\\site-packages torch\\lib\\cudidTT"one of its dependencies

![](https://i-blog.csdnimg.cn/blog_migrate/6289f3be1e8471ff97666c084a6354b6.png)

这就是因为我们batchsize和workers设置太大了的原因

解决方法：

找到train.py的parse\_opt（）函数，将对应batchsize和workers参数调小，如下图：

![](https://i-blog.csdnimg.cn/blog_migrate/7b9ec8004125d392ccd1522d7d338353.png)

（你以为这样就完了吗？No！555~）

接着又会出现下面的错误：

OMP: Hint This means that multiple copies of the OpenMp runtime have been linked into theThat is dangerous, since it can degrade performance or cause incorrect results.. 

![](https://i-blog.csdnimg.cn/blog_migrate/41da2742f6aa246fb1c8b226cb346fb8.png)

 解决方法：

在import os下面加入

```java
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
```

![](https://i-blog.csdnimg.cn/blog_migrate/1d0bd9971fd60a0ae91fea856595c195.png)

（这下没错了吧？你想多了~）

然后又会出现这样的错误：

RuntimeError: resutt type float can't be cast to the desired output type .\_int64 ![](https://i-blog.csdnimg.cn/blog_migrate/2d8a54b2ac89f340b07ebf0f5a760304.png)

 解决方法：

首先进入loss.py文件，将anchors = self.anchors\[i\]改为

```java
anchors, shape = self.anchors[i], p[i].shape
```

![](https://i-blog.csdnimg.cn/blog_migrate/2e4cb99d798e42fc64c49db60da4f14b.png)

↓ ![](https://i-blog.csdnimg.cn/blog_migrate/8cfff1017d638d783162cb423e6d2964.png)

接着往下翻，将indices.append((b, a, gj.clamp\_(o, gain\[3\] - 1), gi.clamp\_(0, gain\[2\] - 1))改为

```java
indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
```

![](https://i-blog.csdnimg.cn/blog_migrate/f74bbc31371e0bb5df17242e57f2fc42.png)

↓ ![](https://i-blog.csdnimg.cn/blog_migrate/16c2b159ccceedf2a43c87419b731909.png)

到这终于能运行啦！撒花✿✿ヽ(°▽°)ノ✿

![](https://i-blog.csdnimg.cn/blog_migrate/4e67b2f8c1ac95ce2cf8503eac928ec6.png)

（训练时错误本来就有很多，但是错误原因网上都能找到的哦~）

到此为止，我们的环境就配好了。

本篇文章是我通过录屏复盘总结的，可能有一些地方有遗忘，大家要是配置过程中有问题还是要看看大佬们的教程（感谢大佬们！） 好了，我这个小白先撤了~下一篇再见啦！

> 本文参考：
> 
> [CUDA与cuDNN安装教程（超详细）\_kylinmin的博客-CSDN博客][CUDA_cuDNN_kylinmin_-CSDN]
> 
> [【零基础上手yolov5】yolov5的安装与相关环境的搭建\_罅隙\`的博客-CSDN博客][yolov5_yolov5_-CSDN]

![](https://i-blog.csdnimg.cn/blog_migrate/406afbeed8f29124a4316812360581bd.gif)


[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501
[Link 1]: #%E5%89%8D%E8%A8%80
[Link 2]: #%E4%B8%80%E3%80%81%E4%BA%86%E8%A7%A3%E6%89%80%E9%9C%80%E9%85%8D%E7%BD%AE
[1.1 CUDA]: #1.1%20CUDA
[1.2 cuDNN]: #1.2%20cuDNN
[1.3 Anconda]: #1.3%20Anconda
[1.4 pycharm]: #1.4%C2%A0pycharm
[1.5 pytorch]: #1.5%20pytorch
[CUDA _cuDNN]: #%E4%BA%8C%E3%80%81%E5%AE%89%E8%A3%85CUDA%20%E5%92%8CcuDNN
[2.1 CUDA]: #2.1%20CUDA%E7%9A%84%E4%B8%8B%E8%BD%BD%E4%B8%8E%E5%AE%89%E8%A3%85
[2.2 cuDNN]: #2.2%20cuDNN%E7%9A%84%E4%B8%8B%E8%BD%BD%E4%B8%8E%E5%AE%89%E8%A3%85
[Anaconda]: #%E4%B8%89%E3%80%81%E5%AE%89%E8%A3%85Anaconda%C2%A0
[pytorch]: #%E5%9B%9B%E3%80%81%E5%AE%89%E8%A3%85pytorch%C2%A0
[YOLOv5 1]: #%E4%BA%94%E3%80%81%E9%85%8D%E7%BD%AEYOLOv5%E7%8E%AF%E5%A2%83
[Link 3]: #%E5%85%AD%E3%80%81%E6%B5%8B%E8%AF%95
[PyTorch]: https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020
[Pytorch_-CSDN]: https://blog.csdn.net/weixin_43334693/category_12186888.html?spm=1001.2014.3001.5482
[https_docs.nvidia.com_cuda_cuda-installation-guide-microsoft-windows_index.html]: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
[Installation Guide _ NVIDIA Deep Learning cuDNN Documentation]: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows
[nvidia]: https://so.csdn.net/so/search?q=nvidia&spm=1001.2101.3001.7020
[_ NVIDIA]: https://www.nvidia.cn/Download/index.aspx?lang=cn
[cuDNN Download _ NVIDIA Developer]: https://developer.nvidia.com/rdp/cudnn-download
[Anaconda3_HowieXue_-CSDN]: https://blog.csdn.net/HowieXue/article/details/118442904?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168076111016800226510971%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168076111016800226510971&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118442904-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=Anconda&spm=1018.2226.3001.4187
[Win11_Pytorch_Pycharm_PyTorch_win11_pytorch]: https://blog.csdn.net/java1314777/article/details/128027977
[mirrors _ ultralytics _ yolov5 _ GitCode]: https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator
[CUDA_cuDNN_kylinmin_-CSDN]: https://blog.csdn.net/anmin8888/article/details/127910084?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168074450216800192219938%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168074450216800192219938&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-4-127910084-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=CUDA&spm=1018.2226.3001.4187
[yolov5_yolov5_-CSDN]: https://blog.csdn.net/whc18858/article/details/127131741?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168068781616800217279306%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168068781616800217279306&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-5-127131741-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE&spm=1018.2226.3001.4187