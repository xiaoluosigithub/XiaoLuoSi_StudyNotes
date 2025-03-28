## 1、先下载好anaconda

## 2、配置环境变量

## 3、打开控制台输入：

~~~bash
conda activate base
~~~

进入到base这个虚拟环境中

## 4、查看你的anaconda对应支持的python版本

因为低版本的anaconda不能支持高版本的python，所以要查看一下你的anaconda最高支持的是哪个版本的python

在base环境中输入：

~~~bash
python --version
~~~

输出的就是你的anaconda最高支持的python版本号

## 5、创建虚拟环境

~~~bash
conda create --name <环境名> python=<版本号>
~~~

版本号不能高于上面输出的哪个版本号，否则可能会出错

## 6、激活虚拟环境

此时你已经创建好了虚拟环境，需要激活进入

~~~bash
conda activate <环境名>
~~~

这样你就进入到了你新创建的conda虚拟环境

## 7、包

### 包的调用范围

在 `base` 环境中安装的包（如 `conda` 或一些其他工具包）是系统范围的工具，通常可以在创建的所有虚拟环境中使用。因此，虚拟环境中的包和工具并不需要在每个环境中都单独安装，只要它们是在 `base` 环境中正确安装并且配置了相关路径，其他虚拟环境是可以调用的。

这也是为什么在 `base` 环境中更新 `conda` 后，不需要在每个虚拟环境中再次更新它的原因。

不过需要注意的是，虚拟环境中的包（如 `numpy`、`pandas` 等）通常是针对具体环境的，因此您需要在每个环境中安装所需的包，确保每个虚拟环境都能运行对应的项目。



下载包有多种方式，讲解一下他们的区别

1. `pip install xxx` 
2. `conda install xxx`

先说这两个:

1. 支持语言:

   - pip 是 python 官方推荐的包下载工具，但是**只能**安装python包
   - conda 是一个**跨平台**（支持linux, mac, win）的`通用包和环境`管理器，它除了支持python外，还能安装各种其他语言的包，例如 C/C++, R语言等

2. 源:

   - pip 从PyPI（Python Package Index）上拉取数据。上面的数据更新更**及时**，涵盖的内容也更加**全面**
   - conda 从 Anaconda.org 上拉取数据。虽然Anaconda上有一些主流Python包，但在数量级上明显少于PyPI，缺少一些小众的包

3. 包的内容:

   - pip 里的软件包为`wheel版`或`源代码发行版`。wheel属于**已编译发新版**的一种，下载好后可以直接使用；而源代码发行版必须要经过**编译**生成可执行程序后才能使用，编译的过程是在用户的机子上进行的

   - conda 里的软件包都是二进制文件，下载后即可使用，不需要经过编译

4. 依赖关系：

   - pip安装包时，尽管也对当前包的依赖做检查，但是**并不保证**当前环境的**所有**包的所有依赖关系都同时满足。当某个环境所安装的包越来越多，产生冲突的可能性就越来越大。
   - conda会检查当前环境下所有包之间的依赖关系，保证当前环境里的**所有**包的所有依赖都会被满足

5. [库的储存位置]：

   - 在conda虚拟环境下使用 pip install 安装的库： 如果使用系统的的python，则库会被保存在 ~/.local/lib/python3.x/site-packages 文件夹中；如果使用的是conda内置的python，则会被保存到 anaconda3/envs/current_env/lib/site-packages中
   - conda install 安装的库都会放在anaconda3/pkgs目录下。这样的好处就是，当在某个环境下已经下载好了某个库，再在另一个环境中还需要这个库时，就可以直接从pkgs目录下将该库复制至新环境而不用重复下载

这样看来,conda下载是不是比pip好一些.

但是使用conda下载有些时候会卡好久(就像下面这样),这时候可以试一试pip下载

~~~bash
Solving environment: |
~~~



还有其他的下载方式如:

mamba : https://www.cnblogs.com/feffery/p/13232119.html

