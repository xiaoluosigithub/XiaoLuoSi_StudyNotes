![](https://i-blog.csdnimg.cn/blog_migrate/0d0d12a231327dfe65f163a57fe60489.gif)

![](https://i-blog.csdnimg.cn/blog_migrate/b3b86bbe9d534b757e204ac8b45d7e99.png)

![962f7cb1b48f44e29d9beb1d499d0530.gif](https://i-blog.csdnimg.cn/blog_migrate/ac3c5d6bfbcbf982e8e9e3632d7f20d1.gif)【YOLOv5改进系列】前期回顾：

[YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析][YOLOv5_0]

[YOLOv5改进系列（1）——添加SE注意力机制][YOLOv5_1_SE]

[YOLOv5改进系列（2）——添加CBAM注意力机制][YOLOv5_2_CBAM]

[YOLOv5改进系列（3）——添加CA注意力机制][YOLOv5_3_CA]

[YOLOv5改进系列（4）——添加ECA注意力机制][YOLOv5_4_ECA]

[YOLOv5改进系列（5）——替换主干网络之 MobileNetV3][YOLOv5_5_ MobileNetV3]

[YOLOv5改进系列（6）——替换主干网络之 ShuffleNetV2][YOLOv5_6_ ShuffleNetV2]

[YOLOv5改进系列（7）——添加SimAM注意力机制][YOLOv5_7_SimAM]

![](https://i-blog.csdnimg.cn/blog_migrate/b4ce96f5d714a9482d2e35bee4cabd02.gif)

目录

[🚀一、SOCA介绍 ][SOCA_]

[1.1 简介][1.1]

[1.2 SAN网络][1.2 SAN]

[1.3.二阶通道注意力（SOCA） ][1.3._SOCA_]

[🚀二、在backbone末端添加SOCA注意力机制方法][backbone_SOCA]

[2.1 添加顺序 ][2.1 _]

[2.2 具体添加步骤 ][2.2 _]

[第①步：在common.py中添加SOCA模块][common.py_SOCA]

[第②步：在yolo.py文件里的parse\_model函数加入类名][yolo.py_parse_model]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 1]

[第⑤步：修改train.py中 ‘--cfg’默认参数][train.py_ _--cfg]

[🚀三、在C3后添加SOCA注意力机制方法][C3_SOCA]

[第③步：创建自定义的yaml文件 ][yaml_]

[第④步：验证是否加入成功][Link 2]

[🌟本人YOLOv5系列导航][YOLOv5]

![](https://i-blog.csdnimg.cn/blog_migrate/0f0d8798ee411fbd1ec5e5da09b6aa42.gif)

## 🚀一、SOCA介绍 

>  *  论文题目：《Second-order Attention Network for Single Image Super-Resolution》
>  *  论文地址：[CVPR19 超分辨率][CVPR19]
>  *  代码地址：[GitHub - daitao/SAN: Second-order Attention Network for Single Image Super-resolution (CVPR-2019)][GitHub - daitao_SAN_ Second-order Attention Network for Single Image Super-resolution _CVPR-2019]

### 1.1 简介 

近年来，深度卷积神经网络（CNN）在单图像超分辨率（SISR）中得到了广泛的研究，并取得了显著的性能。然而，大多数现有的基于CNN的SISR方法主要侧重于更广泛或更深入的架构设计，而忽略了中间层的特征相关性，因此阻碍了CNN的代表能力。为了解决这个问题，论文作者提出了一种二阶注意力网络（SAN），用于更强大的特征表达和特征相关性学习。具体而言，开发了一种新的可训练二阶注意（SOCA）模块，通过使用二阶特征统计来自适应地重新调整信道方向特征，以获得更具鉴别性的表示。

### 1.2 SAN网络 

![](https://i-blog.csdnimg.cn/blog_migrate/579d6bbdf6be7672a3dabb5163651782.png)

从上图中可以看出SAN的主要由四部分组成：

 *  浅层特征提取(shallow feature extraction)即第一个卷积
 *  非局部增强残差组(NLRG) 提取深度特征(deep feature,DF)
 *  上采样模块(upscale module)
 *  重建模块(reconstruction part)即最后一个卷积

### 1.3.二阶通道注意力（SOCA） 

以前大多数基于CNN的SR模型都没有考虑功能的相互依赖性。为了利用这些信息，在CNN中引入了SENet，以重新缩放图像SR的信道特征。然而，SENet仅通过全局平均池利用特征的一阶统计，而忽略高于一阶的统计，从而阻碍了网络的辨别能力。另一方面，最近的研究表明特征的二阶统计分布更有利于获得有区分度的表达，如此才诞生了SOCA。

二阶注意力机制(SOCA)能够更好地学习特征之间的联系，此模块通过利用二阶特征的分布自适应的学习特征的内部依赖关系，SOCA的机制是网络能够专注于更有益的信息且能够提高判别学习的能力。此外，原文提出了一种非局部加强残差组结构能进一步结合非局部操作来提取长程的空间上下文信息。通过堆叠非局部残差组，本文的方法能够利用LR图像的信息且能够忽略低频信息。

## 🚀二、在backbone末端添加SOCA注意力机制方法 

### 2.1 添加顺序 

（1）models/common.py --> 加入新增的网络结构

（2） models/yolo.py --> 设定网络结构的传参细节，将SOCA类名加入其中。（当新的自定义模块中存在输入输出维度时，要使用qw调整输出维度）  
（3） models/yolov5\*.yaml --> 新建一个文件夹，如yolov5s\_SOCA.yaml，修改现有模型结构配置文件。（当引入新的层时，要修改后续的结构中的from参数）  
（4） train.py --> 修改‘--cfg’默认参数，训练时指定模型结构配置文件

### 2.2 具体添加步骤 

#### 第①步：在common.py中添加SOCA模块 

将下面的SOCA代码复制粘贴到common.py文件的末尾

```java
# SOCA moudle 单幅图像超分辨率
from torch.autograd import Function

class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output, der_sacleTrace=None):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

class SOCA(nn.Module):
    # second-order Channel attention
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        cov_mat = CovpoolLayer(x_sub) # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov*x
```

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/7359b7219e09ae487a8f2fdbc6cf021c.png)

#### 第②步：在yolo.py文件里的parse\_model函数加入类名 

首先找到yolo.py里面parse\_model函数的这一行

![](https://i-blog.csdnimg.cn/blog_migrate/bba4391d0c7a836a6dd99b450002d847.png)

然后把刚才加入的类SOCA添加到这个注册表里面：

![](https://i-blog.csdnimg.cn/blog_migrate/e962c831e32ebeabead55ea172dea0f8.png)

或者可以在下面的位置这样加，原理和上面是一样的：

```java
elif m is SOCA:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, *args[1:]]
```

![](https://i-blog.csdnimg.cn/blog_migrate/e274a3f273d1a6b111b88b3b3b9cdc40.png)

解释一下这段代码：

这段是一个判断语句，如果模块 m 在SOCA中，那么就将模块m对应的输入通道数和输出通道数的值分别赋值给 c1 和 c2，然后对 c2进行与之前相同的处理，接下来，将 c1、 c2 以及 args\[1:\] 作为元素，组成新的列表，作为更新后的 args。

#### 第③步：创建自定义的yaml文件 

首先在models文件夹下复制yolov5s.yaml文件，粘贴并重命名为 yolov5s\_SOCA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/7c8d45f59ac882025e7be71a25b171ec.png)

接着修改 yolov5s\_SOCA.yaml，将SOCA模块加到我们想添加的位置。

这里我先介绍第一种，第一种是将SOCA模块放在backbone部分的最末端，这样可以使注意力机制看到整个backbone部分的特征图，将具有全局视野，类似于一个小transformer结构。

将 \[-1，1，SOCA，\[1024\]\]添加到 SPPF 的下一层。即下图中所示位置：

![](https://i-blog.csdnimg.cn/blog_migrate/8607e3adabb28d8be2f24f38cf4dcfc7.png)

同样的下面的head也得修改：

![](https://i-blog.csdnimg.cn/blog_migrate/78edc7ac33a500e4734c8d4c80ea8610.png)

这里我们要把后面两个Concat的from系数分别由\[ − 1 , 14 \] ，\[ − 1 , 10 \]改为\[ − 1 , 15 \]，\[ − 1 , 11 \]。

然后将Detect原始的from系数\[ 17 , 20 , 23 \]要改为\[ 18 , 21 , 24 \] 。

![](https://i-blog.csdnimg.cn/blog_migrate/0b6c22fa91f4a4c4f0c38314179421c5.png)

yolov5s\_SOCA.yaml完整代码：

```java
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license


# Parameters
nc: 20  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone+SE
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
   [-1, 1, SPPF, [1024, 5]],  # 10
   [-1, 1, SOCA,[1024]],
  ]

# YOLOv5 v6.1 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 14

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 18 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 15], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 21 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 11], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 24 (P5/32-large)

   [[18, 21, 24], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

在yolo.py 文件里面配置改为我们刚才自定义的yolov5s\_SOCA.yaml

![](https://i-blog.csdnimg.cn/blog_migrate/593a52d3a7237df3705342119951a9e2.png)

![](https://i-blog.csdnimg.cn/blog_migrate/b159ac68794c304b312c258397815445.png)

然后我们运行yolo.py

![](https://i-blog.csdnimg.cn/blog_migrate/089da237263d7b871fc78e575ad30a84.png)

这样就添加成功啦~

#### 第⑤步：修改train.py中 ‘--cfg’默认参数 

我们先找到 train.py文件的parse\_opt函数，然后将第二行‘--cfg’的 default改为yolov5s\_SOCA.yaml，然后就可以开始训练啦~

![](https://i-blog.csdnimg.cn/blog_migrate/561c876b8b8b1d8d3ddd5347f207350e.png)

## 🚀三、在C3后添加SOCA注意力机制方法 

第二种是将SOCA放在backbone部分每个C3模块的后面，这样可以使注意力机制看到局部的特征，每层进行一次注意力，可以分担学习压力。

步骤和方法1相同，只是yaml文件不同。

所以接下来只放修改yaml文件的部分~

#### 第③步：创建自定义的yaml文件 

将SOCA模块放在每个C3模块的后面，要注意通道的变化。

如下图所示：

![](https://i-blog.csdnimg.cn/blog_migrate/0748a5293dd03bf14ebeb1c63353cf91.png)

同样的，下面的head部分也要做相应的修改:

![](https://i-blog.csdnimg.cn/blog_migrate/7c1876ebfd53e54fce5fb6d6cd689aeb.png)

第二种方法的 yolov5s\_SOCA.yaml 完整代码：

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
   [-1, 3, C3, [128]],
   [-1, 3, SOCA, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 6, C3, [256]],
   [-1, 3, SOCA, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 9, C3, [512]],
   [-1, 3, SOCA, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 10-P5/32
   [-1, 3, C3, [1024]],
   [-1, 3, SOCA, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 13

  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 9], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 17

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 21 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 18], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 24 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 27 (P5/32-large)

   [[21, 24, 27], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

#### 第④步：验证是否加入成功 

同样的方法，我们来运行一下yolo.py 

![](https://i-blog.csdnimg.cn/blog_migrate/bfa4c592030dc698773291750247cd42.png)

OK~收工！

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

![](https://i-blog.csdnimg.cn/blog_migrate/9797652b6577b44685ea4d0f831cb970.gif)


[YOLOv5_0]: https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501
[YOLOv5_1_SE]: https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501
[YOLOv5_2_CBAM]: https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501
[YOLOv5_3_CA]: https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501
[YOLOv5_4_ECA]: https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501
[YOLOv5_5_ MobileNetV3]: https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501
[YOLOv5_6_ ShuffleNetV2]: https://blog.csdn.net/weixin_43334693/article/details/131008642?spm=1001.2014.3001.5501
[YOLOv5_7_SimAM]: https://blog.csdn.net/weixin_43334693/article/details/131031541?spm=1001.2014.3001.5501
[SOCA_]: #%F0%9F%9A%80%E4%B8%80%E3%80%81SOCA%E4%BB%8B%E7%BB%8D%C2%A0
[1.1]: #1.1%20%E7%AE%80%E4%BB%8B
[1.2 SAN]: #1.2%20SAN%E7%BD%91%E7%BB%9C
[1.3._SOCA_]: #1.3.%E4%BA%8C%E9%98%B6%E9%80%9A%E9%81%93%E6%B3%A8%E6%84%8F%E5%8A%9B%EF%BC%88SOCA%EF%BC%89%C2%A0
[backbone_SOCA]: #%F0%9F%9A%80%E4%BA%8C%E3%80%81%E5%9C%A8backbone%E6%9C%AB%E7%AB%AF%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[2.1 _]: #2.1%20%E6%B7%BB%E5%8A%A0%E9%A1%BA%E5%BA%8F%C2%A0
[2.2 _]: #2.2%20%E5%85%B7%E4%BD%93%E6%B7%BB%E5%8A%A0%E6%AD%A5%E9%AA%A4%C2%A0
[common.py_SOCA]: #%E7%AC%AC%E2%91%A0%E6%AD%A5%EF%BC%9A%E5%9C%A8common.py%E4%B8%AD%E6%B7%BB%E5%8A%A0SE%E6%A8%A1%E5%9D%97
[yolo.py_parse_model]: #%C2%A0%E7%AC%AC%E2%91%A1%E6%AD%A5%EF%BC%9A%E5%9C%A8yolo.py%E6%96%87%E4%BB%B6%E9%87%8C%E7%9A%84parse_model%E5%87%BD%E6%95%B0%E5%8A%A0%E5%85%A5%E7%B1%BB%E5%90%8D
[yaml_]: #%E7%AC%AC%E2%91%A2%E6%AD%A5%EF%BC%9A%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E7%9A%84yaml%E6%96%87%E4%BB%B6%C2%A0
[Link 1]: #%C2%A0%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[train.py_ _--cfg]: #%E7%AC%AC%E2%91%A4%E6%AD%A5%EF%BC%9A%E4%BF%AE%E6%94%B9train.py%E4%B8%AD%C2%A0%E2%80%98--cfg%E2%80%99%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0
[C3_SOCA]: #%F0%9F%9A%80%E4%B8%89%E3%80%81%E5%9C%A8C3%E5%90%8E%E6%B7%BB%E5%8A%A0SimAM%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E6%96%B9%E6%B3%95
[Link 2]: #%E7%AC%AC%E2%91%A3%E6%AD%A5%EF%BC%9A%E9%AA%8C%E8%AF%81%E6%98%AF%E5%90%A6%E5%8A%A0%E5%85%A5%E6%88%90%E5%8A%9F
[YOLOv5]: #%F0%9F%8C%9F%E6%9C%AC%E4%BA%BAYOLOv5%E7%B3%BB%E5%88%97%E5%AF%BC%E8%88%AA
[CVPR19]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf
[GitHub - daitao_SAN_ Second-order Attention Network for Single Image Super-resolution _CVPR-2019]: https://github.com/daitao/SAN
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