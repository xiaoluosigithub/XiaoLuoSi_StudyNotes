#### ![](https://i-blog.csdnimg.cn/blog_migrate/29bd6a2fee8eacc7a20a2cbed10c4251.gif) 

## ![](https://i-blog.csdnimg.cn/blog_migrate/04c1a74f62b8ffba43b932fca32456b5.jpeg) 

## 前言 

上一篇我们已经搭建好了YOLOv5的环境（直通车→[YOLOv5入门实践（1）——手把手带你环境配置搭建][YOLOv5_1]），现在就开始第二步利用labelimg标注数据集吧！

![](https://i-blog.csdnimg.cn/blog_migrate/0d226fc1e407fb8b950137a2381d4691.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif) 🍀本人[YOLOv5源码][YOLOv5]详解系列：

[YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析][YOLOv5_1 1]

[YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py][YOLOv5_2_detect.py]

[YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py][YOLOv5_3_train.py]

[YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py][YOLOv5_4_val_test_.py]

[YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml][YOLOv5_5_yolov5s.yaml]

[YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py][YOLOv5_6_1_yolo.py]

[YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py][YOLOv5_7_2_common.py]

## 目录 

[前言][Link 1]

[一、labelimg工具介绍][labelimg]

[二、 labelimg的下载][labelimg 1]

[三、labelimg的安装][labelimg 2]

[四、labelImg 的使用][labelImg]

[4.1 准备工作 ][4.1 _]

[4.2 标注前的设置][4.2]

[4.3 开始标注][4.3]

![](https://i-blog.csdnimg.cn/blog_migrate/9950e362f947a274540a236913789907.gif)

## 一、labelimg工具介绍 

Labelimg是一个图形图像注释工具。

它是用Python编写的，并使用Qt作为其图形界面。

注释以PASCAL VOC格式保存为XML文件，这是使用的ImageNet格式。此外，它还支持YOLO格式和 CreateML 格式。

## 二、 labelimg的下载 

labelimg的下载有两种：

法1：从官网下载→下载地址：[https://github.com/tzutalin/labelImg][https_github.com_tzutalin_labelImg]

![](https://i-blog.csdnimg.cn/blog_migrate/5e6e48cc8f8f4f991fe239be4f18c624.png)

法2：如果你和我一样懒就直接网盘下载吧（感谢提供资源的大佬！）

> 链接：https://pan.baidu.com/s/19GoT4Tb0Mco1STgprxAjPw?pwd=j666  
> 提取码：j666

## 三、labelimg的安装 

第1步：利用cd命令进入labelimg所在的文件夹

```java
d:
```

```java
cd [自己的文件位置]
```

![](https://i-blog.csdnimg.cn/blog_migrate/aed467fbae43e66a41700bc3887e7426.png)

第2步：安装 pyqt，这里我安装的是pyqt5

```java
conda install pyqt=5
```

![](https://i-blog.csdnimg.cn/blog_migrate/d95e9beaaec91ec2ff90ee24e6e52f2e.png)

安装完成就是下图这样：

![](https://i-blog.csdnimg.cn/blog_migrate/ae8e773f2a2eed7373c88eb5520a92b7.png)

第3步：安装完成后，执行命令

```java
pyrcc5 -o libs/resources.py resources.qrc
```

![](https://i-blog.csdnimg.cn/blog_migrate/11fd0ed3844b84324358a2748411b10b.png)

这个命令没有返回结果。

第4步：打开labelimg

```java
python labelImg.py
```

![](https://i-blog.csdnimg.cn/blog_migrate/df173acba743f689951ec5da5816423e.png)

这样就打开了呢~

![](https://i-blog.csdnimg.cn/blog_migrate/14b50e272e1d66a43020083cd6af08f3.png)

## 四、labelImg 的使用 

### 4.1 准备工作 

第1步：在yolov5目录下新建一个名为VOCData的文件夹

（这个是约定俗成，不这么做也行）

![](https://i-blog.csdnimg.cn/blog_migrate/1fd9e2c857635053e219f6d3b2acc6c3.png)

第2步：在VOCData的文件夹内建立Annotations和images文件夹

 *  Annotations：存放标注的标签文件
 *  images：存放需要打标签的图片文件

![](https://i-blog.csdnimg.cn/blog_migrate/04e3fed589eb3200e065e050d81ffe5e.png)

### 4.2 标注前的设置 

将要标注的图片放入images文件夹内，运行软件前可以更改下要标注的类别。这里选了三个类别：花、猫猫和鱼。

![](https://i-blog.csdnimg.cn/blog_migrate/617ebaaa11e3ee08c804b517ad113048.png)

然后我们在labelimg的data文件下找到predefined\_classes.txt 这个txt文档，在里面输入自定义的类别名称，如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/9627896f67ecd1da2e1e2db6cd397bc0.png)

### 4.3 开始标注 

标注前我们先认识一下功能键。如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/34ceb5e06815ea2a183d9e192b80ffef.png)

还有view的一些功能键，如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/e357834b19af6dd1260b9d7b3ccd2b07.png)

常用快捷键如下：

> A：切换到上一张图片
> 
> D：切换到下一张图片
> 
> W：调出标注十字架
> 
> del ：删除标注框框
> 
> Ctrl+u：选择标注的图片文件夹
> 
> Ctrl+r：选择标注好的label标签存在的文件夹

接下来打开图片，按住鼠标左键就可以标注了。

![](https://i-blog.csdnimg.cn/blog_migrate/6ab3d98256fe7ab5c6b1d3ae112f7745.png)

点击鼠标右键还可以移动选框位置和调整大小。

![](https://i-blog.csdnimg.cn/blog_migrate/f390eebbe758f6dafc88f41a4a04d685.png)

标签打完以后可以去Annotations 文件下看到标签文件已经保存在这个目录下。

![](https://i-blog.csdnimg.cn/blog_migrate/f3a3ed18a0e95ec8f086113a191f4c4e.png)

好了，lambelimg的使用就讲到这里啦~

> 本文参考：
> 
> [目标检测---利用labelimg制作自己的深度学习目标检测数据集][---_labelimg]

![](https://i-blog.csdnimg.cn/blog_migrate/465daac376683e725ca34f7da43fe8b5.gif)


[YOLOv5_1]: https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501
[YOLOv5]: https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020
[YOLOv5_1 1]: https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501
[YOLOv5_2_detect.py]: https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501
[YOLOv5_3_train.py]: https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501
[YOLOv5_4_val_test_.py]: https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501
[YOLOv5_5_yolov5s.yaml]: https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501
[YOLOv5_6_1_yolo.py]: https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501
[YOLOv5_7_2_common.py]: https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501
[Link 1]: #%E5%89%8D%E8%A8%80
[labelimg]: #%C2%A0%E4%B8%80%E3%80%81labelimg%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D
[labelimg 1]: #%E4%BA%8C%E3%80%81%C2%A0labelimg%E7%9A%84%E4%B8%8B%E8%BD%BD
[labelimg 2]: #%E4%B8%89%E3%80%81labelimg%E7%9A%84%E5%AE%89%E8%A3%85
[labelImg]: #%E5%9B%9B%E3%80%81labelImg%20%E7%9A%84%E4%BD%BF%E7%94%A8
[4.1 _]: #4.1%20%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C%C2%A0
[4.2]: #4.2%20%E6%A0%87%E6%B3%A8%E5%89%8D%E7%9A%84%E8%AE%BE%E7%BD%AE
[4.3]: #4.3%20%E5%BC%80%E5%A7%8B%E6%A0%87%E6%B3%A8
[https_github.com_tzutalin_labelImg]: https://github.com/tzutalin/labelImg
[---_labelimg]: https://blog.csdn.net/didiaopao/article/details/119808973?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168077368116800182799614%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168077368116800182799614&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-119808973-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=labelimg&spm=1018.2226.3001.4187