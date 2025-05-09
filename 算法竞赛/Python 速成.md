[](https://oi-wiki.org/edit-landing/?ref=/lang/python.md "编辑此页")

Python 速成
=========

关于 Python[](about:blank#%E5%85%B3%E4%BA%8E-python "Permanent link")
-------------------------------------------------------------------

Python 是一门已在世界上广泛使用的解释型语言。它提供了高效的高级数据结构，还能简单有效地面向对象编程，也可以在算法竞赛。

### Python 的优点[](about:blank#python-%E7%9A%84%E4%BC%98%E7%82%B9 "Permanent link")

*   Python 是一门 **解释型** 语言：Python 不需要编译和链接，可以在一定程度上减少操作步骤。
*   Python 是一门 **交互式** 语言：Python 解释器实现了交互式操作，可以直接在终端输入并执行指令。
*   Python **易学易用**：Python 提供了大量的数据结构，也支持开发大型程序。
*   Python **兼容性强**：Python 同时支持 Windows、macOS 和 Unix 操作系统。
*   Python **实用性强**：从简单的输入输出到科学计算甚至于大型 WEB 应用，都可以写出适合的 Python 程序。
*   Python **程序简洁、易读**：Python 代码通常比实现同种功能的其他语言的代码短。
*   Python **支持拓展**：Python 会开发 C 语言程序（即 CPython），支持把 Python 解释器和用 C 语言开发的应用链接，用 Python 扩展和控制该应用。

### 学习 Python 的注意事项[](about:blank#%E5%AD%A6%E4%B9%A0-python-%E7%9A%84%E6%B3%A8%E6%84%8F%E4%BA%8B%E9%A1%B9 "Permanent link")

*   目前主要使用的 Python 版本是 Python 3.7 及以上的版本，Python 2 和 Python 3.6 及以前的 Python 3 已经 [不被支持](https://devguide.python.org/versions/#unsupported-versions)，但仍被一些老旧系统与代码所使用。本文将 **介绍较新版本的 Python**。如果遇到 Python 2 代码，可以尝试 [`2to3`](https://docs.python.org/zh-cn/3/library/2to3.html) 程序将 Python 2 代码转换为 Python 3 代码。
*   Python 的设计理念和语法结构 **与一些其他语言的差异较大**，隐藏了许多底层细节，所以呈现出实用而优雅的风格。
*   Python 是高度动态的解释型语言，因此其 **程序运行速度相对较慢**，尤其在使用其内置的 `for` 循环语句时。在使用 Python 时，应尽量使用 `filter`、`map` 等内置函数，或使用 [列表生成](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 语法的手段来提高程序性能。

环境搭建[](about:blank#%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA "Permanent link")
-------------------------------------------------------------------------

参见 [Python 3](https://oi-wiki.org/tools/compiler/#python-3)。或者：

*   Windows：也可以在 Microsoft Store 中免费而快捷地获取 Python。
    
*   macOS/Linux：通常情况下，大部分的 Linux 发行版中已经自带了 Python。如果只打算学习 Python 语法，并无其它开发需求，不必另外安装 Python。
    
    注意
    
    在一些默认安装（指使用软件包管理器安装）Python 的系统（如 Unix 系统）中，应在终端中运行 `python3` 打开 Python 3 解释器。[1](about:blank#fn:ref1)
    

此外，也可以通过 venv、conda、Nix 等工具管理 Python 工具链和 Python 软件包，创建隔离的虚拟环境，避免出现依赖问题。

作为一种解释型语言，Python 的执行方式和 C++ 有所不同，这种差异在使用 IDE 编程时往往得不到体现，因此这里需要强调一下运行程序的不同方式。

当在命令行中键入 `python3` 或刚刚打开 IDLE 时，你实际进入了一种交互式的编程环境，也称「REPL」（「读取 - 求值 - 输出」循环），初学者可以在这里输入语句并立即看到结果，这让验证一些语法变得极为容易，我们也将在后文中大量使用这种形式。

但若要编写完整的程序，你最好还是新建一个文本文件（通常后缀为 `.py`），然后在命令行中执行 `python3 filename.py`，就能够运行代码看到结果了。

### 一些平台提供的 Python 版本[](about:blank#%E4%B8%80%E4%BA%9B%E5%B9%B3%E5%8F%B0%E6%8F%90%E4%BE%9B%E7%9A%84-python-%E7%89%88%E6%9C%AC "Permanent link")

| 系统名/版本          | python 版本             |
| -------------------- | ----------------------- |
| Noi Linux 2.0        | 3.8.0, Include requests |
| Luogu 评测机         | 3.11.5, NumPy 1.25.2    |
| 基于 Hydro 的 OJ     | 3.8.0+ Include NumPy    |
| Ubuntu 22.04（内置） | 3.10.4                  |
| 微软商店             | 最新正式版              |

注意

本表格在本文撰写时（2025/01/15）时有效，建议前往相关平台重新查证。

目前国内关于 **源码** 的镜像缓存主要是 [北京交通大学自由与开源软件镜像站](https://mirror.bjtu.edu.cn/python/) 和 [华为开源镜像站](https://repo.huaweicloud.com/python/)，可以到那里尝试下载 Python 安装文件。

使用 `pip` 安装第三方库[](about:blank#%E4%BD%BF%E7%94%A8-pip-%E5%AE%89%E8%A3%85%E7%AC%AC%E4%B8%89%E6%96%B9%E5%BA%93 "Permanent link")
-----------------------------------------------------------------------------------------------------------------------------

Python 的生命力很大程度上来自于丰富的第三方库，编写一些实用程序时「调库」是常规操作，`pip` 是首选的安装第三方库的程序。自 Python 3.4 版本起，它被默认包含在 Python 二进制安装程序中。

`pip` 中的第三方库主要存储在 [Python 包索引（PyPI）](https://pypi.org/) 上，用户也可以指定其它第三方库的托管平台。使用方法可参照 [pypi 镜像使用帮助 - 清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) 等使用帮助。你可以在 [MirrorZ](https://mirrorz.org/list/pypi) 上获取更多 PyPI 镜像源。

使用清华大学开源镜像站安装一个包

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span></pre></div></td><td class="code"><div><pre><span></span><code>pip<span class="w"> </span>install<span class="w"> </span>-i<span class="w"> </span>https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple<span class="w"> </span>&lt;some-package&gt;
</code></pre></div></td></tr></tbody></table>

基本语法[](about:blank#%E5%9F%BA%E6%9C%AC%E8%AF%AD%E6%B3%95 "Permanent link")
-------------------------------------------------------------------------

Python 的语法简洁而易懂，也有许多官方和第三方文档与教程。这里仅介绍一些对 OIer 比较实用的语言特性，你可以在 [Python 文档](https://docs.python.org/zh-cn/3/) 和 [Python Wiki](https://wiki.python.org/moin/) 等网页上了解更多关于 Python 的教程。

### 注释[](about:blank#%E6%B3%A8%E9%87%8A "Permanent link")

加入注释并不会对代码的运行产生影响，但加入注释可以使代码更加易懂易用。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="c1"># 用 # 字符开头的是单行注释</span>

<span class="sd">"""</span>
<span class="sd">跨多行字符串会用三引号</span>
<span class="sd">（即三个单引号或三个双引号）</span>
<span class="sd">包裹，但也通常被用于注释</span>
<span class="sd">"""</span>
</code></pre></div></td></tr></tbody></table>

加入注释代码并不会对代码产生影响。我们鼓励加入注释来使代码更加易懂易用。

### 基本数据类型[](about:blank#%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B "Permanent link")

#### 一切皆对象[](about:blank#%E4%B8%80%E5%88%87%E7%9A%86%E5%AF%B9%E8%B1%A1 "Permanent link")

在 Python 中，你无需事先声明变量名及其类型，直接赋值即可创建各种类型的变量：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">x</span> <span class="o">=</span> <span class="o">-</span><span class="mi">3</span>  <span class="c1"># 语句结尾不用加分号</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">x</span>
<span class="go">-3</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">f</span> <span class="o">=</span> <span class="mf">3.1415926535897932384626</span><span class="p">;</span> <span class="n">f</span>  <span class="c1"># 实在想加分号也可以，这里节省了一行</span>
<span class="go">3.141592653589793</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s1</span> <span class="o">=</span> <span class="s2">"O"</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s1</span>  <span class="c1"># 在 Python 中双引号和单引号的作用相同</span>
<span class="go">'O'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">b</span> <span class="o">=</span> <span class="s1">'A'</span> <span class="o">==</span> <span class="mi">65</span>  <span class="c1"># 'A' 和 65 不是一个数据类型，所以不相等</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">b</span>  <span class="c1"># True, False 首字母均大写</span>
<span class="go">False</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kc">True</span> <span class="o">+</span> <span class="mi">1</span> <span class="o">==</span> <span class="mi">2</span> <span class="ow">and</span> <span class="ow">not</span> <span class="kc">False</span> <span class="o">!=</span> <span class="mi">0</span>  <span class="c1"># Python 中的表达式中大多使用单词，但是也支持符号</span>
<span class="go">True</span>
</code></pre></div></td></tr></tbody></table>

但这不代表 Python 没有类型的概念，实际上解释器会根据赋值或运算自动推断变量类型，你可以使用内置函数 `type()` 查看这些变量的类型：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="go">&lt;class 'int'&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
<span class="go">&lt;class 'float'&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">s1</span><span class="p">)</span>  <span class="c1"># 请注意，不要给字符串起名为 str，否则 str 对象会被篡改</span>
<span class="go">&lt;class 'str'&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="n">b</span><span class="p">)</span>
<span class="go">&lt;class 'bool'&gt;</span>
</code></pre></div></td></tr></tbody></table>

[**内置函数**](https://docs.python.org/zh-cn/3/library/functions.html) 是什么？

在 C/C++ 中，很多常用函数都分散在不同的头文件中，但 Python 的解释器内置了许多实用且通用的函数，你可以直接使用而无需注意它们的存在，但这也带来了小问题，这些内置函数的名称多为常见单词，你需要注意避免给自己的变量起相同的名字，否则可能会产生奇怪的结果。

正如我们所看到的，Python 内置有整数、浮点数、字符串和布尔类型，可以类比为 C++ 中的 `int`，`float`，`string` 和 `bool`。但有一些明显的不同之处，比如没有 `char` 字符类型，也没有 `double` 类型（但 `float` 其实对应 C 中的双精度），如果需要更精确的浮点运算，可以使用标准库中的 [decimal](https://docs.python.org/zh-cn/3/library/decimal.html) 模块，如果需要用到复数，Python 还内置了 `complex` 类型（而这也意味着最好不要给变量起名为 `complex`）。 可以看到这些类型都以 `class` 开头，而这正是 Python 不同于 C++ 的关键之处，Python 程序中的所有数据都是由对象或对象间关系来表示的，函数是对象，类型本身也是对象：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
<span class="go">&lt;class 'type'&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="nb">pow</span><span class="p">)</span>  <span class="c1"># 求幂次的内置函数，后文会介绍</span>
<span class="go">&lt;class 'builtin_function_or_method'&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">type</span><span class="p">(</span><span class="nb">type</span><span class="p">)</span>  <span class="c1"># type() 也是内置函数，但有些特殊，感兴趣可自行查阅</span>
<span class="go">&lt;class 'type'&gt;</span>
</code></pre></div></td></tr></tbody></table>

你或许会觉得这些概念一时难以理解且没有用处，所以我们暂时不再深入，在后文的示例中你或许能慢慢体会到，Python 的对象提供了强大的方法，我们在编程时应当优先考虑围绕对象而不是过程进行操作，这会让我们的代码显得更加紧凑明晰。

#### 数字运算[](about:blank#%E6%95%B0%E5%AD%97%E8%BF%90%E7%AE%97 "Permanent link")

有人说，你可以把你系统里装的 Python 当作一个多用计算器，这是事实。  
在交互模式下，你可以在提示符 `>>>` 后面输入一个表达式，就像其他大部分语言（如 C++）一样使用运算符 `+`、`-`、`*`、`/`、`%` 来对数字进行运算，也可以使用 `()` 来进行符合结合律的分组，读者可以自行试验，在这里我们仅展示与 C++ 差异较大的部分：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="mf">5.0</span> <span class="o">*</span> <span class="mi">6</span>  <span class="c1"># 浮点数的运算结果是浮点数</span>
<span class="go">30.0</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">15</span> <span class="o">/</span> <span class="mi">3</span>  <span class="c1"># 与 C/C++ 不同，除法永远返回浮点 float 类型</span>
<span class="go">5.0</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">5</span> <span class="o">/</span> <span class="mi">100000</span>  <span class="c1"># 位数太多，结果显示成科学计数法形式</span>
<span class="go">5e-05</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">5</span> <span class="o">//</span> <span class="mi">3</span> <span class="c1"># 使用整数除法（地板除）则会向下取整，输出整数类型</span>
<span class="go">1</span>
<span class="gp">&gt;&gt;&gt; </span><span class="o">-</span><span class="mi">5</span> <span class="o">//</span> <span class="mi">3</span> <span class="c1"># 符合向下取整原则，注意这与 C/C++ 不同</span>
<span class="go">-2</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">5</span> <span class="o">%</span> <span class="mi">3</span> <span class="c1"># 取模</span>
<span class="go">2</span>
<span class="gp">&gt;&gt;&gt; </span><span class="o">-</span><span class="mi">5</span> <span class="o">%</span> <span class="mi">3</span> <span class="c1"># 负数取模结果一定是非负数，这点也与 C/C++ 不同，不过都满足 (a//b)*b+(a%b)==a </span>
<span class="go">1</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">x</span> <span class="o">=</span> <span class="nb">abs</span><span class="p">(</span><span class="o">-</span><span class="mf">1e4</span><span class="p">)</span>  <span class="c1"># 求绝对值的内置函数</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">x</span> <span class="o">+=</span> <span class="mi">1</span>  <span class="c1"># 没有自增/自减运算符</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">x</span>  <span class="c1"># 科学计数法默认为 float</span>
<span class="go">10001.0</span>
</code></pre></div></td></tr></tbody></table>

在上面的实践中可以发现，除法运算（`/`）永远返回浮点类型（在 Python 2 中返回整数）。如果你想要整数或向下取整的结果的话，可以使用整数除法（`//`）。同样的，你也可以像 C++ 中一样，使用模（`%`）来计算余数，科学计数法的形式也相同。

特别地，Python 用 `**` 即可进行幂运算，还通过内置的 `pow(a, b, mod)` 提供了 [快速幂](https://oi-wiki.org/math/binary-exponentiation/) 的高效实现。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="mi">3</span> <span class="o">**</span> <span class="mi">4</span> <span class="c1"># 幂运算</span>
<span class="go">81</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">2</span> <span class="o">**</span> <span class="mi">512</span>
<span class="go">13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084096</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">pow</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="mi">512</span><span class="p">,</span> <span class="nb">int</span><span class="p">(</span><span class="mf">1e4</span><span class="p">))</span> <span class="c1"># 即 2**512 % 10000 的快速实现, 1e4 是 float 所以要转 int</span>
<span class="go">4096</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">2048</span> <span class="o">**</span> <span class="mi">2048</span> <span class="c1"># 在IDLE里试试大整数？</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mf">0.1</span> <span class="o">+</span> <span class="mf">0.1</span> <span class="o">+</span> <span class="mf">0.1</span> <span class="o">-</span> <span class="mf">0.3</span> <span class="o">==</span> <span class="mf">0.</span>  <span class="c1"># 和 C/C++ 一样需要注意浮点数不能直接判相等</span>
<span class="go">False</span>
</code></pre></div></td></tr></tbody></table>

#### 数据类型判断[](about:blank#%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%E5%88%A4%E6%96%AD "Permanent link")

对于一个变量，可以使用 `type(object)` 返回变量的类型，例如 `type(8)` 和 `type('a')` 的值分别为 `<class 'int'>` 和 `<class 'str'>`。

#### [基本输入输出](https://docs.python.org/zh-cn/3/tutorial/inputoutput.html)[](about:blank#%E5%9F%BA%E6%9C%AC%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA "Permanent link")

Python 中的输入输出主要通过内置函数 `input()` 和 `print()` 完成，`print()` 的用法十分符合直觉：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">a</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">];</span> <span class="nb">print</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>  <span class="c1"># 打印时默认末尾换行</span>
<span class="go">3</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">ans</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">ans</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>  <span class="c1"># 可以输出任意多个变量，默认以空格间隔</span>
<span class="go">1 2</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">a</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">end</span><span class="o">=</span><span class="s1">''</span><span class="p">)</span>  <span class="c1"># 令 end='', 使末尾不换行</span>
<span class="go">1 2&gt;&gt;&gt;</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">a</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">sep</span><span class="o">=</span><span class="s1">', '</span><span class="p">)</span>  <span class="c1"># 令 sep=', '，改变间隔样式</span>
<span class="go">1, 2</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span> <span class="o">+</span> <span class="s1">', '</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">a</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>  <span class="c1"># 输出同上，但是手动拼接成一整个字符串</span>
</code></pre></div></td></tr></tbody></table>

`input()` 函数的行为接近 C++ 中的 `getline()`，即将一整行作为字符串读入，且末尾没有换行符。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="nb">input</span><span class="p">(</span><span class="s1">'请输入一串数字: '</span><span class="p">);</span> <span class="n">s</span>  <span class="c1"># 自己调试时可以向 input() 传入字符串作为提示</span>
<span class="go">请输入一串数字: 1 2 3 4 5 6</span>
<span class="go">'1 2 3 4 5 6'</span>
</code></pre></div></td></tr></tbody></table>

#### 字符串[](about:blank#%E5%AD%97%E7%AC%A6%E4%B8%B2 "Permanent link")

Python 3 提供了强大的基于 [Unicode](https://docs.python.org/zh-cn/3/howto/unicode.html#unicode-howto) 的字符串类型，使用起来和 C++ 中的 `string` 类似，一些概念如转义字符也都相通，除了加号拼接和索引访问，还额外支持数乘 `*` 重复字符串，和 `in` 操作符。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">s1</span> <span class="o">=</span> <span class="s2">"O"</span>  <span class="c1"># 单引号和双引号都能包起字符串，有时可节省转义字符</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s1</span> <span class="o">+=</span> <span class="s1">'I-Wiki'</span>  <span class="c1"># 为和 C++ 同步建议使用双引号 </span>
<span class="gp">&gt;&gt;&gt; </span><span class="s1">'OI'</span> <span class="ow">in</span> <span class="n">s1</span>  <span class="c1"># 检测子串很方便</span>
<span class="go">True</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">len</span><span class="p">(</span><span class="n">s1</span><span class="p">)</span>  <span class="c1"># 类似 C++ 的 s.length()，但更通用</span>
<span class="go">7</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s2</span> <span class="o">=</span> <span class="s2">""" 感谢你的阅读</span>
<span class="gp">... </span><span class="s2">欢迎参与贡献!</span>
<span class="go">"""   # 使用三重引号的字符串可以跨越多行</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s2">s1 + s2 </span>
<span class="go">'OI-Wiki 感谢你的阅读\n欢迎参与贡献!'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s2">print(s1 + s2)  # 这里使用了 print() 函数打印字符串</span>
<span class="go">OI-Wiki 感谢你的阅读</span>
<span class="go">欢迎参与贡献!</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s2">s2[2] * 2 + s2[3] + s2[-1]  # 负数索引从右开始计数，加上 len(s)，相当于模 n 的剩余类环</span>
<span class="go">'谢谢你!'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s2">s1[0] = 'o'  # str 是不可变类型，不能原地修改，其实 += 也是创建了新的对象</span>
<span class="gt">Traceback (most recent call last):</span>
  File <span class="nb">"&lt;stdin&gt;"</span>, line <span class="m">1</span>, in <span class="n">&lt;module&gt;</span>
<span class="gr">TypeError</span>: <span class="n">'str' object does not support item assignment</span>
</code></pre></div></td></tr></tbody></table>

Python 支持多种复合数据类型，可将不同值组合在一起。最常用的 `list`，类型是用方括号标注、逗号分隔的一组值。例如，`[1, 2, 3]` 和 `['a','b','c']` 都是列表。

除了索引，字符串还支持_切片_，它的设计非常精妙，格式为 `s[左闭索引:右开索引:步长]`：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="s1">'OI-Wiki 感谢你的阅读</span><span class="se">\n</span><span class="s1">欢迎参与贡献!'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="p">[:</span><span class="mi">8</span><span class="p">]</span>  <span class="c1"># 省略左闭索引则从头开始</span>
<span class="go">'OI-Wiki '</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="p">[</span><span class="mi">8</span><span class="p">:</span><span class="mi">14</span><span class="p">]</span>  <span class="c1"># 左闭右开设计的妙处，长度为 14-8=6，还和上一个字符串无缝衔接</span>
<span class="go">'感谢你的阅读'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="p">[</span><span class="o">-</span><span class="mi">4</span><span class="p">:]</span>  <span class="c1"># 省略右开索引则直到结尾</span>
<span class="go">'与贡献!'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="p">[</span><span class="mi">8</span><span class="p">:</span><span class="mi">14</span><span class="p">:</span><span class="mi">2</span><span class="p">]</span>  <span class="c1"># 步长为2</span>
<span class="go">'感你阅'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>  <span class="c1"># 步长为 -1 时，获得了反转的字符串</span>
<span class="go">'!献贡与参迎欢\n读阅的你谢感 ikiW-IO'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span>  <span class="c1"># 但原来的字符串并未改变</span>
<span class="go">'OI-Wiki 感谢你的阅读\n欢迎参与贡献!'</span>
</code></pre></div></td></tr></tbody></table>

在最新的 Python 3 版本中，字符串是以 Unicode 编码的，也就是说，Python 的字符串支持多语言。[2](about:blank#fn:ref2)在 Python 中，可以对一个 Unicode 字符使用内置函数 `ord()` 将其转换为对应的 Unicode 编码，逆向的转换使用内置函数 `chr()`。C/C++ 中 `char` 类型也可以和 对应的 ASCII 码互转。

如果想把数字转换成对应的字符串，可以使用内置函数 `str()`，反之可以使用 `int()` 和 `float()`，你可以类比为 C/C++ 中的强制类型转换，但括号不是加在类型上而是作为函数的一部分括住参数。

Python 的字符串类型提供了许多强大的方法，包括计算某字符的索引与出现次数，转换大小写等等，这里就不一一列举，强烈建议查看 [官方文档](https://docs.python.org/zh-cn/3/library/stdtypes.html#text-sequence-type-str) 熟悉常用方法，遇到字符串操作应当首先考虑使用这些方法而非自力更生。

### 创建数组[](about:blank#%E5%88%9B%E5%BB%BA%E6%95%B0%E7%BB%84 "Permanent link")

从 C++ 转过来的同学可能很迷惑怎么在 Python 中创建数组，这里就介绍在 Python 开「数组」的语法，需要强调我们介绍的其实是几种 [序列类型](https://docs.python.org/zh-cn/3/library/stdtypes.html#iterator-types)，和 C 的数组有着本质区别，而更接近 C++ 中的 `vector`。

#### 使用 `list`[](about:blank#%E4%BD%BF%E7%94%A8-list "Permanent link")

列表（`list`）大概是 Python 中最常用也最强大的序列类型，列表中可以存放任意类型的元素，包括嵌套的列表，这符合数据结构中「广义表」的定义。请注意不要将其与 C++ STL 中的双向链表 [`list`](https://oi-wiki.org/csl/sequence-container/#list) 混淆，故本文将使用「列表」而非 `list` 以免造成误解。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span>
<span class="normal">26</span>
<span class="normal">27</span>
<span class="normal">28</span>
<span class="normal">29</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="p">[]</span>  <span class="c1"># 创建空列表，注意列表使用方括号</span>
<span class="go">[]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">8</span><span class="p">,</span> <span class="mi">13</span><span class="p">];</span> <span class="n">nums</span>  <span class="c1"># 初始化列表，注意整个列表可以直接打印</span>
<span class="go">[0, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span> <span class="n">nums</span>  <span class="c1"># 支持索引访问，支持修改元素</span>
<span class="go">[1, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">nums</span><span class="p">[</span><span class="o">-</span><span class="mi">2</span><span class="p">]</span><span class="o">+</span><span class="n">nums</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">]);</span> <span class="n">nums</span>  <span class="c1"># append() 同 vector 的 push_back()，也都没有返回值</span>
<span class="go">[1, 1, 2, 3, 5, 8, 13, 21]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">pop</span><span class="p">()</span>  <span class="c1"># 弹出并返回末尾元素，可以当栈使用；其实还可指定位置，默认是末尾</span>
<span class="go">21</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">insert</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">);</span> <span class="n">nums</span>  <span class="c1"># 同 vector 的 insert(position, val)</span>
<span class="go">[1, 1, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">remove</span><span class="p">(</span><span class="mi">1</span><span class="p">);</span> <span class="n">nums</span>  <span class="c1"># 按值移除元素（只删第一个出现的），若不存在则抛出错误</span>
<span class="go">[1, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">len</span><span class="p">(</span><span class="n">nums</span><span class="p">)</span>  <span class="c1"># 求列表长度，类似 vector 的 size()，但 len() 是内置函数</span>
<span class="go">7</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">reverse</span><span class="p">();</span> <span class="n">nums</span>  <span class="c1"># 原地逆置</span>
<span class="go">[13, 8, 5, 3, 2, 1, 1]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">sorted</span><span class="p">(</span><span class="n">nums</span><span class="p">)</span>  <span class="c1"># 获得排序后的列表</span>
<span class="go">[1, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span>  <span class="c1"># 但原来的列表并未排序</span>
<span class="go">[13, 8, 5, 3, 2, 1, 1]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">sort</span><span class="p">();</span> <span class="n">nums</span>  <span class="c1"># 原地排序，可以指定参数 key 作为排序标准</span>
<span class="go">[1, 1, 2, 3, 5, 8, 13]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">count</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># 类似 std::count()</span>
<span class="go">2</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># 返回值首次出现项的索引号，若不存在则抛出错误</span>
<span class="go">0</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span><span class="o">.</span><span class="n">clear</span><span class="p">();</span> <span class="n">nums</span>  <span class="c1"># 同 vector 的 clear()</span>
</code></pre></div></td></tr></tbody></table>

以上示例展现了列表与 `vector` 的相似之处，`vector` 中常用的操作一般也都能在列表中找到对应方法，不过某些方法如 `len()`,`sorted()` 会以内置函数的面目出现，而 STL 算法中的函数如 `find()`,`count()`,`max_element()`,`sort()`,`reverse()` 在 Python 中又成了对象的方法，使用时需要注意区分，更多方法请参见官方文档的 [列表详解](https://docs.python.org/zh-cn/3/tutorial/datastructures.html#more-on-lists)。下面将展示列表作为 Python 的基本序列类型的一些强大功能：

Python 支持多种复合数据类型，可将不同值组合在一起。最常用的 `list`，类型是用方括号标注、逗号分隔的一组值。例如，`[1, 2, 3]` 和 `['a','b','c']` 都是列表。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="s1">'1'</span><span class="p">]</span> <span class="o">+</span> <span class="p">[</span><span class="s2">"2"</span><span class="p">,</span> <span class="mf">3.0</span><span class="p">]</span>  <span class="c1"># 列表直接相加生成一个新列表</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span>  <span class="c1"># 这里存放不同的类型只是想说明可以这么做，但这不是好的做法</span>
<span class="go">[1, '1', '2', 3.0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="mi">3</span> <span class="ow">in</span> <span class="n">lst</span>  <span class="c1"># 实用的成员检测操作，字符串也有该操作且还支持子串检测</span>
<span class="go">True</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="s1">'1'</span><span class="p">]</span> <span class="ow">in</span> <span class="n">lst</span>  <span class="c1"># 仅支持单个成员检测，不会发现「子序列」</span>
<span class="go">False</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span><span class="p">[</span><span class="mi">1</span><span class="p">:</span><span class="mi">3</span><span class="p">]</span> <span class="o">=</span> <span class="p">[</span><span class="mi">2</span><span class="p">,</span> <span class="mi">3</span><span class="p">];</span> <span class="n">lst</span>  <span class="c1"># 切片并赋值，原列表被修改</span>
<span class="go">[1, 2, 3, 3.0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>  <span class="c1"># 获得反转后的新列表</span>
<span class="go">[3.0, 3, 2, 1]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span> <span class="o">*=</span> <span class="mi">2</span><span class="p">;</span> <span class="n">lst</span>  <span class="c1"># 数乘拼接</span>
<span class="go">[1, 2, 3, 3.0, 1, 2, 3, 3.0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="k">del</span> <span class="n">lst</span><span class="p">[</span><span class="mi">4</span><span class="p">:];</span> <span class="n">lst</span>  <span class="c1"># 也可写 lst[4:] = []，del 语句不止可以用于删除序列中元素</span>
<span class="go">[1, 2, 3, 3.0]</span>
</code></pre></div></td></tr></tbody></table>

以上示例展现了列表作为序列的一些常用操作，可以看出许多操作如切片是与字符串相通的，但字符串是「不可变序列」而列表是「可变序列」，故可以通过切片灵活地修改列表。在 C/C++ 中我们往往会通过循环处理字符数组，下面将展示如何使用 [「列表推导式」](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) 在字符串和列表之间转换：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="c1"># 建立一个 [65, 70) 区间上的整数数组，range 也是一种类型，可看作左闭右开区间，第三个参数为步长可省略</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nums</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="mi">65</span><span class="p">,</span><span class="mi">70</span><span class="p">))</span>  <span class="c1"># 记得 range 外面还要套一层 list()</span>
<span class="go">[65, 66, 67, 68, 69]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span> <span class="o">=</span> <span class="p">[</span><span class="nb">chr</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">nums</span><span class="p">]</span>  <span class="c1"># 列表推导式的典型结构，[exp for var in iterable if cond]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">lst</span>  <span class="c1"># 上两句可以合并成 [chr(x) for x in range(65,70)]</span>
<span class="go">['A', 'B', 'C', 'D', 'E']</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="s1">''</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">lst</span><span class="p">);</span> <span class="n">s</span> <span class="c1"># 用空字符串 '' 拼接列表中的元素生成新字符串</span>
<span class="go">'ABCDE'</span>
<span class="go">&gt;&gt; list(s)  # 字符串生成字符列表</span>
<span class="go">['A', 'B', 'C', 'D', 'E']</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># 如果你不知道有 s.lower() 方法就可能写出下面这样新瓶装旧酒的表达式</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s1">''</span><span class="o">.</span><span class="n">join</span><span class="p">([</span><span class="nb">chr</span><span class="p">(</span><span class="nb">ord</span><span class="p">(</span><span class="n">ch</span><span class="p">)</span> <span class="o">-</span> <span class="mi">65</span> <span class="o">+</span> <span class="mi">97</span><span class="p">)</span> <span class="k">for</span> <span class="n">ch</span> <span class="ow">in</span> <span class="n">s</span> <span class="k">if</span> <span class="n">ch</span> <span class="o">&gt;=</span> <span class="s1">'A'</span> <span class="ow">and</span> <span class="n">ch</span> <span class="o">&lt;=</span> <span class="s1">'Z'</span><span class="p">])</span>  
<span class="go">'abcde'</span>
</code></pre></div></td></tr></tbody></table>

下面演示一些在 OI 中更常见的场景，比如二维「数组」：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span>
<span class="normal">26</span>
<span class="normal">27</span>
<span class="normal">28</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">vis</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">3</span><span class="p">]</span> <span class="o">*</span> <span class="mi">3</span>  <span class="c1"># 开一个 3*3 的全 0 数组</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis</span> 
<span class="go">[[0, 0, 0], [0, 0, 0], [0, 0, 0]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span> <span class="n">vis</span>  <span class="c1"># 怎么会把其他行也修改了？</span>
<span class="go">[[1, 0, 0], [1, 0, 0], [1, 0, 0]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># 先来看下一维列表的赋值</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">];</span> <span class="n">a2</span> <span class="o">=</span> <span class="n">a1</span><span class="p">;</span> <span class="n">a3</span> <span class="o">=</span> <span class="n">a1</span><span class="p">[:]</span>  <span class="c1"># 列表也可以直接被赋给新的变量</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span><span class="p">;</span> <span class="n">a1</span>  <span class="c1"># 修改列表 a1，似乎正常</span>
<span class="go">[1, 0, 0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a2</span>  <span class="c1"># 怎么 a2 也被改变了</span>
<span class="go">[1, 0, 0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a3</span>  <span class="c1"># a3 没有变化</span>
<span class="go">[0, 0, 0]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">id</span><span class="p">(</span><span class="n">a1</span><span class="p">)</span> <span class="o">==</span> <span class="nb">id</span><span class="p">(</span><span class="n">a2</span><span class="p">)</span> <span class="ow">and</span> <span class="nb">id</span><span class="p">(</span><span class="n">a1</span><span class="p">)</span> <span class="o">!=</span> <span class="nb">id</span><span class="p">(</span><span class="n">a3</span><span class="p">)</span>  <span class="c1"># 内置函数 id() 给出对象的「标识值」，可类比为地址，地址相同说明是一个对象</span>
<span class="go">True</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis2</span> <span class="o">=</span> <span class="n">vis</span><span class="p">[:]</span>  <span class="c1"># 拷贝一份二维列表</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span> <span class="o">=</span> <span class="mi">2</span><span class="p">;</span> <span class="n">vis</span>  <span class="c1"># vis 会被批量修改</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis2</span>  <span class="c1"># 但 vis2 是切片拷贝还是被改了</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">id</span><span class="p">(</span><span class="n">vis</span><span class="p">)</span> <span class="o">!=</span> <span class="nb">id</span><span class="p">(</span><span class="n">vis2</span><span class="p">)</span>  <span class="c1"># vis 和 vis2 不是一个对象</span>
<span class="go">True</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># vis2 虽然不是 vis 的引用，但其中对应行都指向相同的对象</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[</span><span class="nb">id</span><span class="p">(</span><span class="n">vis</span><span class="p">[</span><span class="n">i</span><span class="p">])</span> <span class="o">==</span> <span class="nb">id</span><span class="p">(</span><span class="n">vis2</span><span class="p">[</span><span class="n">i</span><span class="p">])</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">)]</span>
<span class="go">[True, True, True]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># 回看二维列表自身</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[</span><span class="nb">id</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">vis</span><span class="p">]</span>  <span class="c1"># 具体数字和这里不一样但三个值一定相同，说明是三个相同对象</span>
<span class="go">[139760373248192, 139760373248192, 139760373248192]</span>
</code></pre></div></td></tr></tbody></table>

其实有一个重要的事实，Python 中赋值只传递了引用而非创建新值，你可以创建不同类型的变量并赋给新变量，验证发现二者的标识值是相同的，只不过直到现在我们才介绍了列表这一种可变类型，而给数字、字符串这样的不可变类型赋新值时实际上创建了新的对象，故而前后两个变量互不干扰。但列表是可变类型，所以我们修改一个列表的元素时，另一个列表由于指向同一个对象所以也被修改了。创建二维数组也是类似的情况，示例中用乘法创建二维列表相当于把 `[0]*3` 这个一维列表重复了 3 遍，所以涉及其中一个列表的操作会同时影响其他两个列表。更不幸的是，在将二维列表赋给其他变量的时候，就算用切片来拷贝，也只是「浅拷贝」，其中的元素仍然指向相同的对象，解决这个问题需要使用标准库中的 [`deepcopy`](https://docs.python.org/3/library/copy.html)，或者尽量避免整个赋值二维列表。不过还好，创建二维列表时避免创建重复的列表还是比较简单，只需使用「列表推导式」：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">vis1</span> <span class="o">=</span> <span class="p">[[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">3</span> <span class="k">for</span> <span class="n">_</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">)]</span>  <span class="c1"># 把用不到的循环计数变量设为下划线 _ 是一种惯例</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># 但在 REPL 中 _ 默认指代上一个表达式输出的结果，故也可使用双下划线</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis1</span>
<span class="go">[[0, 0, 0], [0, 0, 0], [0, 0, 0]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="p">[</span><span class="nb">id</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">vis1</span><span class="p">]</span>  <span class="c1"># 具体数字和这里不一样但三个值一定不同，说明是三个不同对象</span>
<span class="go">[139685508981248, 139685508981568, 139685508981184]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">vis1</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
<span class="go">[[1, 0, 0], [0, 0, 0], [0, 0, 0]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a2</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">10</span>  <span class="c1"># 访问和赋值二维数组</span>
</code></pre></div></td></tr></tbody></table>

我们未讲循环的用法就先介绍了列表推导式，这是由于 Python 是高度动态的解释型语言，因此其程序运行有大量的额外开销。尤其是 **for 循环在 Python 中运行的奇慢无比**。因此在使用 Python 时若想获得高性能，尽量使用使用列表推导式，或者 `filter`,`map` 等内置函数直接操作整个序列来避免循环，当然这还是要根据具体问题而定。

#### 使用 NumPy[](about:blank#%E4%BD%BF%E7%94%A8-numpy "Permanent link")

什么是 NumPy

[NumPy](https://numpy.org/) 是著名的 Python 科学计算库，提供高性能的数值及矩阵运算。在测试算法原型时可以利用 NumPy 避免手写排序、求最值等算法。NumPy 的核心数据结构是 `ndarray`，即 n 维数组，它在内存中连续存储，是定长的。此外 NumPy 核心是用 C 编写的，运算效率很高。不过需要注意，它不是标准库的一部分，可以使用 `pip install numpy` 安装，但不保证 OI 考场环境中可用（参见文首 [Python 版本](about:blank#%E4%B8%80%E4%BA%9B%E5%B9%B3%E5%8F%B0%E6%8F%90%E4%BE%9B%E7%9A%84-python-%E7%89%88%E6%9C%AC)）。

下面的代码将介绍如何利用 NumPy 建立多维数组并进行访问。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span>
<span class="normal">26</span>
<span class="normal">27</span>
<span class="normal">28</span>
<span class="normal">29</span>
<span class="normal">30</span>
<span class="normal">31</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>  <span class="c1"># 请自行搜索 import 的意义和用法</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">empty</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span> <span class="c1"># 开容量为 3 的空数组，注意没有初始化为 0</span>
<span class="go">array([0.00000000e+000, 0.00000000e+000, 2.01191014e+180])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span> <span class="c1"># 开 3*3 的数组，并初始化为 0</span>
<span class="go">array([[0., 0., 0.],</span>
<span class="go">       [0., 0., 0.],</span>
<span class="go">       [0., 0., 0.]])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">),</span> <span class="n">dtype</span><span class="o">=</span><span class="nb">int</span><span class="p">)</span> <span class="c1"># 开 3×3 的整数数组</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span> <span class="c1"># 访问和赋值</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span> <span class="c1"># 更友好的语法</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="o">.</span><span class="n">shape</span> <span class="c1"># 数组的形状</span>
<span class="go">(3, 3)</span>

<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="p">[:</span><span class="mi">2</span><span class="p">,</span> <span class="p">:</span><span class="mi">2</span><span class="p">]</span> <span class="c1"># 取前两行、前两列构成的子阵，无拷贝</span>
<span class="go">array([[1, 0],</span>
<span class="go">       [0, 0]])</span>

<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="p">[:,</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">]]</span> <span class="c1"># 获取第 1、3 列，无拷贝</span>
<span class="go">array([[1, 0],</span>
<span class="go">       [0, 0],</span>
<span class="go">       [0, 0]])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">max</span><span class="p">(</span><span class="n">a1</span><span class="p">)</span> <span class="c1"># 获取数组最大值</span>
<span class="go">1</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="o">.</span><span class="n">flatten</span><span class="p">()</span> <span class="c1"># 将数组展平</span>
<span class="go">array([1, 0, 0, 0, 0, 0, 0, 0, 0])</span>

<span class="gp">&gt;&gt;&gt; </span><span class="n">np</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">a1</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span> <span class="c1"># 沿行方向对数组进行排序，返回排序结果</span>
<span class="go">array([[0, 0, 1],</span>
<span class="go">       [0, 0, 0],</span>
<span class="go">       [0, 0, 0]])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a1</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span> <span class="c1"># 沿行方向对数组进行原地排序</span>
</code></pre></div></td></tr></tbody></table>

#### 使用 `array`[](about:blank#%E4%BD%BF%E7%94%A8-array "Permanent link")

[`array`](https://docs.python.org/zh-cn/3/library/array.html) 是 Python 标准库提供的一种高效数值数组，可以紧凑地表示基本类型值的数组，但不支持数组嵌套，也很少见到有人使用它，这里只是顺便提一下。

若无特殊说明，后文出现「数组」一般指「列表」。

### [输入输出](https://docs.python.org/zh-cn/3/tutorial/inputoutput.html)[](about:blank#%E8%BE%93%E5%85%A5%E8%BE%93%E5%87%BA "Permanent link")

Python 中的输入输出主要通过内置函数 `input()` 和 `print()` 完成。前文已经介绍过，下面介绍进阶用法。

#### 格式化输出[](about:blank#%E6%A0%BC%E5%BC%8F%E5%8C%96%E8%BE%93%E5%87%BA "Permanent link")

算法竞赛中通常只涉及到基本的数值和字符串输出，`print()` 已基本足够，只有当涉及到浮点数位数时需要用到格式化字符串输出。格式化有三种方法，第一种也是最老旧的方法是使用 `printf()` 风格的 `%` 操作符；另一种是利用 [`format` 函数](https://docs.python.org/3/library/string.html#formatstrings)；第三种是 Python 3.6 新增的 [f-string](https://docs.python.org/zh-cn/3/tutorial/inputoutput.html#formatted-string-literals)，最为简洁，但不保证考场中的 Python 版本足够新。详细丰富的说明可以参考 [这个网页](https://www.python-course.eu/python3_formatted_output.php)，尽管更推荐使用 `format()` 方法，但为了获得与 C 接近的体验，下面仅演示与 `printf()` 类似的老式方法：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">pi</span> <span class="o">=</span> <span class="mf">3.1415926</span><span class="p">;</span> <span class="nb">print</span><span class="p">(</span><span class="s1">'</span><span class="si">%.4f</span><span class="s1">'</span> <span class="o">%</span> <span class="n">pi</span><span class="p">)</span>   <span class="c1"># 格式为 %[flags][width][.precision]type</span>
<span class="go">3.1416</span>
<span class="gp">&gt;&gt;&gt; </span><span class="s1">'</span><span class="si">%.4f</span><span class="s1"> - </span><span class="si">%8f</span><span class="s1"> = </span><span class="si">%d</span><span class="s1">'</span> <span class="o">%</span> <span class="p">(</span><span class="n">pi</span><span class="p">,</span> <span class="mf">0.1416</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>  <span class="c1"># 右边多个参数用 () 括住，后面会看到其实是「元组」 </span>
<span class="go">'3.1416 - 0.141600 = 3'</span>
</code></pre></div></td></tr></tbody></table>

#### `split()` 函数[](about:blank#split-%E5%87%BD%E6%95%B0 "Permanent link")

`input()` 函数的行为接近 C++ 中的 `getline()`，即将一整行作为字符串读入，且末尾没有换行符，但在算法竞赛中，常见的输入形式是一行输入多个数值，因此就需要使用字符串的 `split()` 方法并搭配列表推导式得到存放数值类型的列表，下面以输入 n 个数求平均值为例演示输入 n 个数得到「数组」的方法：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="nb">input</span><span class="p">(</span><span class="s1">'请输入一串数字: '</span><span class="p">);</span> <span class="n">s</span>  <span class="c1"># 自己调试时可以向 input() 传入字符串作为提示</span>
<span class="go">请输入一串数字: 1 2 3 4 5 6</span>
<span class="go">'1 2 3 4 5 6'</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a</span> <span class="o">=</span> <span class="n">s</span><span class="o">.</span><span class="n">split</span><span class="p">();</span> <span class="n">a</span>
<span class="go">['1', '2', '3', '4', '5', '6']</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">a</span> <span class="o">=</span> <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">a</span><span class="p">];</span> <span class="n">a</span>
<span class="go">[1, 2, 3, 4, 5, 6]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="c1"># 以上输入过程可写成一行 a = [int(x) for x in input().split()]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">sum</span><span class="p">(</span><span class="n">a</span><span class="p">)</span> <span class="o">/</span> <span class="nb">len</span><span class="p">(</span><span class="n">a</span><span class="p">)</span>  <span class="c1"># sum() 是内置函数</span>
<span class="go">3.5</span>
</code></pre></div></td></tr></tbody></table>

有时题目会在每行输入固定几个数，比如边的起点、终点、权重，如果只用上面提到的方法就只能每次读入数组然后根据下标赋值，这时可以使用 Python 的「拆包」特性一次赋值多个变量：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span>
<span class="go">1 2 4</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">u</span><span class="p">,</span><span class="n">v</span><span class="p">,</span><span class="n">w</span><span class="p">)</span>
<span class="go">1 2 4</span>
</code></pre></div></td></tr></tbody></table>

题目中经常遇到输入 N 行的情况，可我们还没有讲最基本的循环语句，但 Python 强大的序列操作能在不使用循环的情况下应对多行输入，下面假设将各条边的起点、终点、权值分别读入三个数组：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">N</span> <span class="o">=</span> <span class="mi">4</span><span class="p">;</span> <span class="n">mat</span> <span class="o">=</span> <span class="p">[[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>
<span class="go">1 3 3 </span>
<span class="go">1 4 1 </span>
<span class="go">2 3 4 </span>
<span class="go">3 4 1 </span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">mat</span>  <span class="c1"># 先按行读入二维数组</span>
<span class="go">[[1, 3, 3], [1, 4, 1], [2, 3, 4], [3, 4, 1]]</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="nb">list</span><span class="p">,</span> <span class="nb">zip</span><span class="p">(</span><span class="o">*</span><span class="n">mat</span><span class="p">))</span>   
<span class="go"># *将 mat 解包得到里层的多个列表</span>
<span class="go"># zip() 将多个列表中对应元素聚合成元组，得到一个迭代器</span>
<span class="go"># map(list, iterable) 将序列中的元素（这里为元组）转成列表</span>
<span class="gp">&gt;&gt;&gt; </span><span class="nb">print</span><span class="p">(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span>  <span class="c1"># 直接将 map() 得到的迭代器拆包，分别赋值给 u, v, w</span>
<span class="go">[1, 1, 2, 3] [3, 4, 3, 4] [3, 1, 4, 1]</span>
</code></pre></div></td></tr></tbody></table>

上述程序实际上相当于先读入一个 N 行 3 列的矩阵，然后将其转置成 3 行 N 列的矩阵，也就是外层列表中嵌套了 3 个列表，最后将代表这起点、终点、权值的 3 个列表分别赋值给 u, v, w。内置函数 [`zip()`](https://docs.python.org/zh-cn/3/library/functions.html#zip) 可以将多个等长序列中的对应元素拼接在「元组」内，得到新序列。而 `map()` 其实是函数式编程的一种操作，它将一个给定函数作用于 `zip()` 所产生序列的元素，这里就是用 `list()` 将元组变成列表。你可以自行练习使用 `*` 和 [`zip()`](https://docs.python.org/zh-cn/3/library/functions.html#zip)，[`map()`](https://docs.python.org/zh-cn/3/library/functions.html#map) 以理解其含义。需要注意的是 Python 3 中 `zip()` 和 `map()` 创建的不再返回列表而是返回迭代器，这里暂不解释它们之间的异同，你可以认为迭代器可以产生列表中的各个元素，用 `list()` 套住迭代器就能生成列表。

### [控制流程](https://docs.python.org/zh-cn/3/tutorial/controlflow.html)[](about:blank#%E6%8E%A7%E5%88%B6%E6%B5%81%E7%A8%8B "Permanent link")

尽管我们已经学习了 Python 的许多特性，但到目前为止我们展示的 Python 代码都是单行语句，这掩盖了 Python 和 C 在代码风格上的重大差异：首先，Python 中不用 `{}` 而是用缩进表示块结构，如果缩进没有对齐会直接报错，如果 tab 和 空格混用也会报错；其次，块结构开始的地方比如 `if` 和 `for` 语句的行末要有冒号 `:`。这有助于代码的可读性，但你也可能怀念 C 那种自由的体验，毕竟如果复制粘贴时因为丢失缩进而不得不手动对齐是很恼人的。

#### 循环结构[](about:blank#%E5%BE%AA%E7%8E%AF%E7%BB%93%E6%9E%84 "Permanent link")

列表推导式能在一行内高效地完成批量操作，但有时为了压行我们已经显得过分刻意，许多场景下还是只能使用循环结构，所以我们再以读入多行数据为例展示 Python 中的循环是如何编写的：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="c1"># 请注意从现在开始我们不再使用 REPL，请自行复制多行数据</span>
<span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="p">([]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">3</span><span class="p">))</span>  <span class="c1"># 多变量赋值</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">4</span><span class="p">):</span>  <span class="c1"># 这里假设输入 4 行数据</span>
    <span class="n">_u</span><span class="p">,</span> <span class="n">_v</span><span class="p">,</span> <span class="n">_w</span> <span class="o">=</span> <span class="p">[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span>
    <span class="n">u</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">_u</span><span class="p">),</span> <span class="n">v</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">_v</span><span class="p">),</span> <span class="n">w</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">_w</span><span class="p">)</span>
    <span class="c1"># 不可进行类似 cin &gt;&gt; u[i] &gt;&gt; v[i] &gt;&gt; w[i] 的操作，因为必定超出列表当前的长度</span>
    <span class="c1"># 当然你可以选择初始化长度为 MAXN 的全 0 列表，不过需要记住真实长度并删掉多余元素</span>
<span class="nb">print</span><span class="p">(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

需要注意，Python 中的 for 循环和 C/C++ 有较大的差别，其作用类似 C++ 11 引入的 [「基于范围的循环」](https://oi-wiki.org/new/#%E5%9F%BA%E4%BA%8E%E8%8C%83%E5%9B%B4%E7%9A%84-for-%E5%BE%AA%E7%8E%AF)，实质是迭代序列中的元素，比如编写循环遍历数组下标需要迭代 `range(len(lst))`，而非真正定义起始和终止条件，所以使用起来并没有 C/C++ 灵活。

下面再用 while 循环展示行数不定的情况下如何输入：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[],</span> <span class="p">[]</span>  <span class="c1"># 多变量赋值，其实同上</span>
<span class="n">s</span> <span class="o">=</span> <span class="nb">input</span><span class="p">()</span>  <span class="c1"># 注意 Python 中赋值语句不能放在条件表达式中</span>
<span class="k">while</span> <span class="n">s</span><span class="p">:</span>  <span class="c1"># 不能像 C 那样 while(!scanf())</span>
    <span class="c1"># 用切片拼接避免了 append()，注意列表推导式中又嵌套了列表</span>
    <span class="n">u</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">u</span><span class="p">)</span> <span class="p">:],</span> <span class="n">v</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">v</span><span class="p">)</span> <span class="p">:],</span> <span class="n">w</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">w</span><span class="p">)</span> <span class="p">:]</span> <span class="o">=</span> <span class="p">[[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)]</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">s</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span>
    <span class="n">s</span> <span class="o">=</span> <span class="nb">input</span><span class="p">()</span>
<span class="c1"># Python 3.8 引入了 walrus operator 海象运算符后，你可以节省两行，但考场环境很可能不支持</span>
<span class="k">while</span> <span class="n">s</span> <span class="o">:=</span> <span class="nb">input</span><span class="p">():</span>
    <span class="n">u</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">u</span><span class="p">)</span> <span class="p">:],</span> <span class="n">v</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">v</span><span class="p">)</span> <span class="p">:],</span> <span class="n">w</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">w</span><span class="p">)</span> <span class="p">:]</span> <span class="o">=</span> <span class="p">[[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)]</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">s</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span>
<span class="nb">print</span><span class="p">(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

#### 选择结构[](about:blank#%E9%80%89%E6%8B%A9%E7%BB%93%E6%9E%84 "Permanent link")

和 C/C++ 大同小异，一些形式上的差别都在下面的示例中有所展示，此外还需注意条件表达式中不允许使用赋值运算符（Python 3.8 以上可用 [`:=`](https://www.python.org/dev/peps/pep-0572/)），以及 [没有 switch 语句](https://docs.python.org/zh-cn/3/faq/design.html#why-isn-t-there-a-switch-or-case-statement-in-python)。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="c1"># 条件表达式两侧无括号</span>
<span class="k">if</span> <span class="mi">4</span> <span class="o">&gt;=</span> <span class="mi">3</span> <span class="o">&gt;</span> <span class="mi">2</span> <span class="ow">and</span> <span class="mi">3</span> <span class="o">!=</span> <span class="mi">5</span> <span class="o">==</span> <span class="mi">5</span> <span class="o">!=</span> <span class="mi">7</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"关系运算符可以连续使用"</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="kc">None</span> <span class="ow">or</span> <span class="p">[]</span> <span class="ow">or</span> <span class="o">-</span><span class="mi">2</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"&amp;&amp;  ||  !"</span><span class="p">,</span> <span class="s2">"与  或  非"</span><span class="p">,</span> <span class="s2">"and or not"</span><span class="p">,</span> <span class="n">sep</span><span class="o">=</span><span class="s2">"</span><span class="se">\n</span><span class="s2">"</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"善用 and/or 可节省行数"</span><span class="p">)</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">x</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"负数也是 True，不执行本句"</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">x</span> <span class="o">&amp;</span> <span class="mi">1</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"用 elif 而不是 else if</span><span class="se">\n</span><span class="s2">"</span> <span class="s2">"位运算符与 C 相近，偶数&amp;1 得 0，不执行本句"</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"也有三目运算符"</span><span class="p">)</span> <span class="k">if</span> <span class="n">x</span> <span class="k">else</span> <span class="nb">print</span><span class="p">(</span><span class="s2">"注意结构"</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

#### 异常处理[](about:blank#%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86 "Permanent link")

尽管 C++ 中有 [try 块](https://zh.cppreference.com/w/cpp/language/try_catch) 用于异常处理，但竞赛中一般从不使用，而 Python 中常见的是 [EAFP](https://docs.python.org/zh-cn/3/glossary.html#term-eafp) 风格，故而代码中可能大量使用 [`try-except`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#the-try-statement) 语句，在后文介绍 `dict` 这一结构时还会用到，这里展示：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="n">s</span> <span class="o">=</span> <span class="s2">"OI-wiki"</span>
<span class="n">pat</span> <span class="o">=</span> <span class="s2">"NOIP"</span>
<span class="n">x</span> <span class="o">=</span> <span class="n">s</span><span class="o">.</span><span class="n">find</span><span class="p">(</span><span class="n">pat</span><span class="p">)</span>  <span class="c1"># find() 找不到返回 -1</span>
<span class="k">try</span><span class="p">:</span>
    <span class="n">y</span> <span class="o">=</span> <span class="n">s</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="n">pat</span><span class="p">)</span>  <span class="c1"># index() 找不到则抛出错误</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>  <span class="c1"># 这句被跳过</span>
<span class="k">except</span> <span class="ne">ValueError</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"没找到"</span><span class="p">)</span>
    <span class="k">try</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>  <span class="c1"># 此时 y 并没有定义，故又会抛出错误</span>
    <span class="k">except</span> <span class="ne">NameError</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"无法输出 y"</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"原因:"</span><span class="p">,</span> <span class="n">e</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

#### [文件读写](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement)[](about:blank#%E6%96%87%E4%BB%B6%E8%AF%BB%E5%86%99 "Permanent link")

Python 内置函数 [`open()`](https://docs.python.org/3/library/functions.html#open) 用于文件读写，为了防止读写过程中出错导致文件未被正常关闭，这里只介绍使用 [`with`](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement) 语句的安全读写方法：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="n">a</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="s2">"in.txt"</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">N</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">f</span><span class="o">.</span><span class="n">readline</span><span class="p">())</span>  <span class="c1"># 读入第一行的 N</span>
    <span class="n">a</span><span class="p">[</span><span class="nb">len</span><span class="p">(</span><span class="n">a</span><span class="p">)</span> <span class="p">:]</span> <span class="o">=</span> <span class="p">[[</span><span class="nb">int</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">f</span><span class="o">.</span><span class="n">readline</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">()]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>

<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="s2">"out.txt"</span><span class="p">,</span> <span class="s2">"w"</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">f</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="s2">"1</span><span class="se">\n</span><span class="s2">"</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

关于文件读写的函数有很多，分别适用于不同的场景，由于 OI 赛事尚不支持使用 Python，这里从略。

### 内置容器[](about:blank#%E5%86%85%E7%BD%AE%E5%AE%B9%E5%99%A8 "Permanent link")

Python 内置了许多强大的容器类型，只有熟练使用并了解其特点才能真正让 Python 在算法竞赛中有用武之地，除了上面详细介绍的 `list`（列表），还有 `tuple`（元组）、[`dict`](https://docs.python.org/zh-cn/3/library/stdtypes.html#mapping-types-dict)（字典）和 `set`（集合）这几种类型。

元组可以简单理解成不可变的列表，不过还需注意「不可变」的内涵，如果元组中的某元素是可变类型比如列表，那么仍可以修改该列表的值，元组中存放的是对列表的引用所以元组本身并没有改变。元组的优点是开销较小且「[可哈希](https://docs.python.org/zh-cn/3/glossary.html)」，后者在创建字典和集合时非常有用。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="n">tup</span> <span class="o">=</span> <span class="nb">tuple</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">],</span> <span class="mi">4</span><span class="p">])</span>  <span class="c1"># 由列表得到元组</span>
<span class="c1"># 等同于 tup = ([1,2], 4)</span>
<span class="n">tup</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">tup</span><span class="p">)</span>
<span class="n">a</span><span class="p">,</span> <span class="n">b</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span> <span class="s2">"I-Wiki"</span>  <span class="c1"># 多变量赋值其实是元组拆包</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">id</span><span class="p">(</span><span class="n">a</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">b</span><span class="p">))</span>
<span class="n">b</span><span class="p">,</span> <span class="n">a</span> <span class="o">=</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span>
<span class="nb">print</span><span class="p">(</span><span class="nb">id</span><span class="p">(</span><span class="n">a</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">b</span><span class="p">))</span>  <span class="c1"># 你应该会看到 a, b 的 id 值现在互换了</span>
<span class="c1"># 这更说明 Python 中，变量更像是名字，赋值只是让其指代对象</span>
</code></pre></div></td></tr></tbody></table>

字典就像 C++ STL 中的 [`map`](https://oi-wiki.org/csl/associative-container/#map)（请注意和 Python 中内置函数 [`map()`](https://docs.python.org/zh-cn/3/library/functions.html#map) 区分）用于存储键值对，形式类似 [JSON](https://docs.python.org/3/library/json.html)，但 JSON 中键必须是字符串且以双引号括住，字典则更加灵活强大，可哈希的对象都可作为字典的键。需要注意 Python 几次版本更新后字典的特性有了较多变化，包括其中元素的顺序等，请自行探索。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="n">dic</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"key"</span><span class="p">:</span> <span class="s2">"value"</span><span class="p">}</span>  <span class="c1"># 基本形式</span>
<span class="n">dic</span> <span class="o">=</span> <span class="p">{</span><span class="nb">chr</span><span class="p">(</span><span class="n">i</span><span class="p">):</span> <span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">65</span><span class="p">,</span> <span class="mi">91</span><span class="p">)}</span>  <span class="c1"># 大写字母到对应 ASCII 码的映射，注意断句</span>
<span class="n">dic</span> <span class="o">=</span> <span class="nb">dict</span><span class="p">(</span><span class="nb">zip</span><span class="p">([</span><span class="nb">chr</span><span class="p">(</span><span class="n">i</span><span class="p">)</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">65</span><span class="p">,</span> <span class="mi">91</span><span class="p">)],</span> <span class="nb">range</span><span class="p">(</span><span class="mi">65</span><span class="p">,</span> <span class="mi">91</span><span class="p">)))</span>  <span class="c1"># 效果同上</span>
<span class="n">dic</span> <span class="o">=</span> <span class="p">{</span><span class="n">dic</span><span class="p">[</span><span class="n">k</span><span class="p">]:</span> <span class="n">k</span> <span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="n">dic</span><span class="p">}</span>  <span class="c1"># 将键值对逆转，for k in dic 迭代其键</span>
<span class="n">dic</span> <span class="o">=</span> <span class="p">{</span><span class="n">v</span><span class="p">:</span> <span class="n">k</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="n">dic</span><span class="o">.</span><span class="n">items</span><span class="p">()}</span>  <span class="c1"># 和上行作用相同，dic.items() 以元组存放单个键值对</span>
<span class="n">dic</span> <span class="o">=</span> <span class="p">{</span>
    <span class="n">k</span><span class="p">:</span> <span class="n">v</span> <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">v</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">dic</span><span class="o">.</span><span class="n">items</span><span class="p">(),</span> <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="o">-</span><span class="n">x</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="p">}</span>  <span class="c1"># 字典按值逆排序，用到了 lambda 表达式</span>

<span class="nb">print</span><span class="p">(</span><span class="n">dic</span><span class="p">[</span><span class="s2">"A"</span><span class="p">])</span>  <span class="c1"># 返回 dic 中 以 'A' 为键的项，这里值为65</span>
<span class="n">dic</span><span class="p">[</span><span class="s2">"a"</span><span class="p">]</span> <span class="o">=</span> <span class="mi">97</span>  <span class="c1"># 将 d[key] 设为 value，字典中原无 key 就是直接插入</span>
<span class="k">if</span> <span class="s2">"b"</span> <span class="ow">in</span> <span class="n">dic</span><span class="p">:</span>  <span class="c1"># LBYL(Look Before You Leap) 风格</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">dic</span><span class="p">[</span><span class="s2">"b"</span><span class="p">])</span>  <span class="c1"># 若字典中无该键则会出错，故先检查</span>
<span class="k">else</span><span class="p">:</span>
    <span class="n">dic</span><span class="p">[</span><span class="s2">"b"</span><span class="p">]</span> <span class="o">=</span> <span class="mi">98</span>

<span class="c1"># 经典场景 统计出现次数</span>
<span class="c1"># 新键不存在于原字典，需要额外处理</span>
<span class="k">try</span><span class="p">:</span>  <span class="c1"># EAFP (Easier to Ask for Forgiveness than Permission) 风格</span>
    <span class="n">cnter</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">+=</span> <span class="mi">1</span>
<span class="k">except</span> <span class="ne">KeyError</span><span class="p">:</span>
    <span class="n">cnter</span><span class="p">[</span><span class="n">key</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
</code></pre></div></td></tr></tbody></table>

集合就像 C++ STL 中的 [`set`](https://oi-wiki.org/csl/associative-container/#set)，不会保存重复的元素，可以看成只保存键的字典。需要注意集合和字典都用 `{}` 括住，不过单用 `{}` 会创建空字典而不是空集合，这里就不再给出示例。

### 编写函数[](about:blank#%E7%BC%96%E5%86%99%E5%87%BD%E6%95%B0 "Permanent link")

Python 中定义函数无需指定参数类型和返回值类型，无形中为 OI 选手减少了代码量

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">def</span> <span class="nf">add</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">a</span> <span class="o">+</span> <span class="n">b</span>  <span class="c1"># 动态类型的优势，a 和 b 也可以是字符串</span>


<span class="k">def</span> <span class="nf">add_no_swap</span><span class="p">(</span><span class="n">a</span><span class="p">,</span> <span class="n">b</span><span class="p">):</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"in func #1:"</span><span class="p">,</span> <span class="nb">id</span><span class="p">(</span><span class="n">a</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">b</span><span class="p">))</span>
    <span class="n">a</span> <span class="o">+=</span> <span class="n">b</span>
    <span class="n">b</span><span class="p">,</span> <span class="n">a</span> <span class="o">=</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"in func #2:"</span><span class="p">,</span> <span class="nb">id</span><span class="p">(</span><span class="n">a</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">b</span><span class="p">))</span>  <span class="c1"># a, b 已交换</span>
    <span class="k">return</span> <span class="n">a</span><span class="p">,</span> <span class="n">b</span>  <span class="c1"># 返回多个值，其实就是返回元组，可以拆包接收</span>


<span class="n">lst1</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">]</span>
<span class="n">lst2</span> <span class="o">=</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"outside func #1:"</span><span class="p">,</span> <span class="nb">id</span><span class="p">(</span><span class="n">lst1</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">lst2</span><span class="p">))</span>
<span class="n">add_no_swap</span><span class="p">(</span><span class="n">lst1</span><span class="p">,</span> <span class="n">lst2</span><span class="p">)</span>
<span class="c1"># 函数外 lst1, lst2 并未交换</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"outside func #2:"</span><span class="p">,</span> <span class="nb">id</span><span class="p">(</span><span class="n">lst1</span><span class="p">),</span> <span class="nb">id</span><span class="p">(</span><span class="n">lst2</span><span class="p">))</span>
<span class="c1"># 不过值确实已经改变</span>
<span class="nb">print</span><span class="p">(</span><span class="n">lst1</span><span class="p">,</span> <span class="n">lst2</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

#### 默认参数[](about:blank#%E9%BB%98%E8%AE%A4%E5%8F%82%E6%95%B0 "Permanent link")

Python 中函数的参数非常灵活，有关键字参数、可变参数等，但在算法竞赛中这些特性的用处并不是很大，这里只介绍一下默认参数，因为 C++ 中也有默认参数，且在 Python 中使用默认参数很有可能遇到坑。

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">def</span> <span class="nf">append_to</span><span class="p">(</span><span class="n">element</span><span class="p">,</span> <span class="n">to</span><span class="o">=</span><span class="p">[]):</span>
    <span class="n">to</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">element</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">to</span>


<span class="n">lst1</span> <span class="o">=</span> <span class="n">append_to</span><span class="p">(</span><span class="mi">12</span><span class="p">)</span>
<span class="n">lst2</span> <span class="o">=</span> <span class="n">append_to</span><span class="p">(</span><span class="mi">42</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">lst1</span><span class="p">,</span> <span class="n">lst2</span><span class="p">)</span>

<span class="c1"># 你可能以为输出是 [12] [42]</span>
<span class="c1"># 但运行结果其实是 [12] [12, 42]</span>


<span class="c1"># 这是因为默认参数的值仅仅在函数定义的时候赋值一次</span>
<span class="c1"># 默认参数的值应该是不可变对象，使用 None 占位是一种最佳实践</span>
<span class="k">def</span> <span class="nf">append_to</span><span class="p">(</span><span class="n">element</span><span class="p">,</span> <span class="n">to</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">to</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
        <span class="n">to</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">to</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">element</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">to</span>
</code></pre></div></td></tr></tbody></table>

#### 类型标注[](about:blank#%E7%B1%BB%E5%9E%8B%E6%A0%87%E6%B3%A8 "Permanent link")

Python 是一个动态类型检查的语言，以灵活但隐式的方式处理类型，Python 解释器仅仅在运行时检查类型是否正确，并且允许在运行时改变变量类型，俗话说「动态类型一时爽，代码重构火葬场」，程序中的一些错误可能在运行时才会暴露：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="k">if</span> <span class="kc">False</span><span class="p">:</span>
<span class="gp">... </span>    <span class="mi">1</span> <span class="o">+</span> <span class="s2">"two"</span>  <span class="c1"># This line never runs, so no TypeError is raised</span>
<span class="gp">... </span><span class="k">else</span><span class="p">:</span>
<span class="gp">... </span>    <span class="mi">1</span> <span class="o">+</span> <span class="mi">2</span>
<span class="gp">...</span>
<span class="go">3</span>

<span class="gp">&gt;&gt;&gt; </span><span class="mi">1</span> <span class="o">+</span> <span class="s2">"two"</span>  <span class="c1"># Now this is type checked, and a TypeError is raised</span>
<span class="go">TypeError: unsupported operand type(s) for +: 'int' and 'str'</span>
</code></pre></div></td></tr></tbody></table>

Python 3.5 后引入了类型标注，允许设置函数参数和返回值的类型，但只是作为提示，并没有实际的限制作用，需要静态检查工具才能排除这类错误（例如 [PyCharm](https://www.jetbrains.com/pycharm/) 和 [Mypy](http://mypy-lang.org/)），所以显得有些鸡肋，对于 OIer 来说更是只需了解，可按如下方式对函数的参数和返回值设置类型标注：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span>
<span class="normal">9</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">def</span> <span class="nf">headline</span><span class="p">(</span>
    <span class="n">text</span><span class="p">,</span>  <span class="c1"># type: str</span>
    <span class="n">width</span><span class="o">=</span><span class="mi">80</span><span class="p">,</span>  <span class="c1"># type: int</span>
    <span class="n">fill_char</span><span class="o">=</span><span class="s2">"-"</span><span class="p">,</span>  <span class="c1"># type: str</span>
<span class="p">):</span>  <span class="c1"># type: (...) -&gt; str</span>
    <span class="k">return</span> <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">text</span><span class="o">.</span><span class="n">title</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span><span class="o">.</span><span class="n">center</span><span class="p">(</span><span class="n">width</span><span class="p">,</span> <span class="n">fill_char</span><span class="p">)</span>


<span class="nb">print</span><span class="p">(</span><span class="n">headline</span><span class="p">(</span><span class="s2">"type comments work"</span><span class="p">,</span> <span class="n">width</span><span class="o">=</span><span class="mi">40</span><span class="p">))</span>
</code></pre></div></td></tr></tbody></table>

除了函数参数，变量也是可以类型标注的，你可以通过调用 `__annotations__` 来查看函数中所有的类型标注。变量类型标注赋予了 Python 静态语言的性质，即声明与赋值分离：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="gp">&gt;&gt;&gt; </span><span class="n">nothing</span><span class="p">:</span> <span class="nb">str</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">nothing</span>
<span class="go">NameError: name 'nothing' is not defined</span>

<span class="gp">&gt;&gt;&gt; </span><span class="vm">__annotations__</span>
<span class="go">{'nothing': &lt;class 'str'&gt;}</span>
</code></pre></div></td></tr></tbody></table>

装饰器[](about:blank#%E8%A3%85%E9%A5%B0%E5%99%A8 "Permanent link")
---------------------------------------------------------------

装饰器是一个函数，接受一个函数或方法作为其唯一的参数，并返回一个新函数或方法，其中整合了修饰后的函数或方法，并附带了一些额外的功能。简而言之，可以在不修改函数代码的情况下，增加函数的功能。相关知识可以参考 [官方文档](https://docs.python.org/3/glossary.html#term-decorator)。

部分装饰器在竞赛中非常实用，比如 [`lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache)，可以为函数自动增加记忆化的能力，在递归算法中非常实用：

`@lru_cache(maxsize=128,typed=False)`

*   传入的参数有 2 个：`maxsize` 和 `typed`，如果不传则 `maxsize` 的默认值为 128，`typed` 的默认值为 `False`。
*   其中 `maxsize` 参数表示的是 LRU 缓存的容量，即被装饰的方法的最大可缓存结果的数量。如果该参数值为 128，则表示被装饰方法最多可缓存 128 个返回结果；如果 `maxsize` 传入为 `None` 则表示可以缓存无限个结果。
*   如果 `typed` 设置为 `True`，不同类型的函数参数将被分别缓存，例如，`f(3)` 和 `f(3.0)` 会缓存两次。

以下是使用 `lru_cache` 优化计算斐波那契数列的例子：

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="nd">@lru_cache</span><span class="p">(</span><span class="n">maxsize</span><span class="o">=</span><span class="kc">None</span><span class="p">)</span>
<span class="k">def</span> <span class="nf">fib</span><span class="p">(</span><span class="n">n</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">n</span> <span class="o">&lt;</span> <span class="mi">2</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">n</span>
    <span class="k">return</span> <span class="n">fib</span><span class="p">(</span><span class="n">n</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span> <span class="o">+</span> <span class="n">fib</span><span class="p">(</span><span class="n">n</span> <span class="o">-</span> <span class="mi">2</span><span class="p">)</span>
</code></pre></div></td></tr></tbody></table>

常用内置库[](about:blank#%E5%B8%B8%E7%94%A8%E5%86%85%E7%BD%AE%E5%BA%93 "Permanent link")
-----------------------------------------------------------------------------------

在这里介绍一些写算法可能用得到的内置库，具体用法可以自行搜索或者阅读 [官方文档](https://docs.python.org/3/library/index.html)。

| 库名                                                         | 用途                         |
| ------------------------------------------------------------ | ---------------------------- |
| [`array`](https://docs.python.org/3/library/array.html)      | 定长数组                     |
| [`argparse`](https://docs.python.org/3/library/argparse.html) | 命令行参数处理               |
| [`bisect`](https://docs.python.org/3/library/bisect.html)    | 二分查找                     |
| [`collections`](https://docs.python.org/3/library/collections.html) | 有序字典、双端队列等数据结构 |
| [`fractions`](https://docs.python.org/3/library/fractions.html) | 有理数                       |
| [`heapq`](https://docs.python.org/3/library/heapq.html)      | 基于堆的优先级队列           |
| [`io`](https://docs.python.org/3/library/io.html)            | 文件流、内存流               |
| [`itertools`](https://docs.python.org/3/library/itertools.html) | 迭代器                       |
| [`math`](https://docs.python.org/3/library/math.html)        | 数学函数                     |
| [`os.path`](https://docs.python.org/3/library/os.html)       | 系统路径等                   |
| [`random`](https://docs.python.org/3/library/random.html)    | 随机数                       |
| [`re`](https://docs.python.org/3/library/re.html)            | 正则表达式                   |
| [`struct`](https://docs.python.org/3/library/struct.html)    | 转换结构体和二进制数据       |
| [`sys`](https://docs.python.org/3/library/sys.html)          | 系统信息                     |

从例题对比 C++ 与 Python[](about:blank#%E4%BB%8E%E4%BE%8B%E9%A2%98%E5%AF%B9%E6%AF%94-c-%E4%B8%8E-python "Permanent link")
-------------------------------------------------------------------------------------------------------------------

[例题 洛谷 P4779【模板】单源最短路径（标准版）](https://www.luogu.com.cn/problem/P4779)

给定一个 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "n(1 \leq n \leq 10^5)") 个点、![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "m(1 \leq m \leq 2\times 10^5)") 条有向边的带非负权图，请你计算从 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "s") 出发，到每个点的距离。数据保证能从 ![](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7 "s") 出发到任意点。

### 声明常量[](about:blank#%E5%A3%B0%E6%98%8E%E5%B8%B8%E9%87%8F "Permanent link")

C++Python

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cstdio&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cstring&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;queue&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;vector&gt;</span>
<span class="k">using</span><span class="w"> </span><span class="k">namespace</span><span class="w"> </span><span class="nn">std</span><span class="p">;</span>
<span class="k">constexpr</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">N</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mf">1e5</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">5</span><span class="p">,</span><span class="w"> </span><span class="n">M</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mf">2e5</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">5</span><span class="p">;</span>
</code></pre></div></td></tr></tbody></table>

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal">1</span>
<span class="normal">2</span>
<span class="normal">3</span>
<span class="normal">4</span>
<span class="normal">5</span>
<span class="normal">6</span>
<span class="normal">7</span>
<span class="normal">8</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">try</span><span class="p">:</span>  <span class="c1"># 引入优先队列模块</span>
    <span class="kn">import</span> <span class="nn">Queue</span> <span class="k">as</span> <span class="nn">pq</span>  <span class="c1"># python version &lt; 3.0</span>
<span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span>
    <span class="kn">import</span> <span class="nn">queue</span> <span class="k">as</span> <span class="nn">pq</span>  <span class="c1"># python3.*</span>

<span class="n">N</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mf">1e5</span> <span class="o">+</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">M</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mf">2e5</span> <span class="o">+</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">INF</span> <span class="o">=</span> <span class="mh">0x3F3F3F3F</span>
</code></pre></div></td></tr></tbody></table>

### 声明前向星结构体和其它变量[](about:blank#%E5%A3%B0%E6%98%8E%E5%89%8D%E5%90%91%E6%98%9F%E7%BB%93%E6%9E%84%E4%BD%93%E5%92%8C%E5%85%B6%E5%AE%83%E5%8F%98%E9%87%8F "Permanent link")

C++Python

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">struct</span><span class="w"> </span><span class="nc">qxx</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kt">int</span><span class="w"> </span><span class="n">nex</span><span class="p">,</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">;</span>
<span class="p">};</span>

<span class="n">qxx</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">M</span><span class="p">];</span>
<span class="kt">int</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">N</span><span class="p">],</span><span class="w"> </span><span class="n">cnt</span><span class="p">;</span>

<span class="kt">void</span><span class="w"> </span><span class="nf">add_path</span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">f</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">v</span><span class="p">)</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="o">++</span><span class="n">cnt</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">qxx</span><span class="p">{</span><span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">],</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">},</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">cnt</span><span class="p">;</span><span class="w"> </span><span class="p">}</span>

<span class="k">using</span><span class="w"> </span><span class="n">pii</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">pair</span><span class="o">&lt;</span><span class="kt">int</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="o">&gt;</span><span class="p">;</span>
<span class="n">priority_queue</span><span class="o">&lt;</span><span class="n">pii</span><span class="p">,</span><span class="w"> </span><span class="n">vector</span><span class="o">&lt;</span><span class="n">pii</span><span class="o">&gt;</span><span class="p">,</span><span class="w"> </span><span class="n">greater</span><span class="o">&lt;</span><span class="n">pii</span><span class="o">&gt;&gt;</span><span class="w"> </span><span class="n">q</span><span class="p">;</span>
<span class="kt">int</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">N</span><span class="p">];</span>
</code></pre></div></td></tr></tbody></table>

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">class</span> <span class="nc">qxx</span><span class="p">:</span>  <span class="c1"># 前向星类（结构体）</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">nex</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">t</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">v</span> <span class="o">=</span> <span class="mi">0</span>


<span class="n">e</span> <span class="o">=</span> <span class="p">[</span><span class="n">qxx</span><span class="p">()</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">M</span><span class="p">)]</span>  <span class="c1"># 链表</span>
<span class="n">h</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>
<span class="n">cnt</span> <span class="o">=</span> <span class="mi">0</span>

<span class="n">dist</span> <span class="o">=</span> <span class="p">[</span><span class="n">INF</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>
<span class="n">q</span> <span class="o">=</span> <span class="n">pq</span><span class="o">.</span><span class="n">PriorityQueue</span><span class="p">()</span>  <span class="c1"># 定义优先队列，默认第一元小根堆</span>


<span class="k">def</span> <span class="nf">add_path</span><span class="p">(</span><span class="n">f</span><span class="p">,</span> <span class="n">t</span><span class="p">,</span> <span class="n">v</span><span class="p">):</span>  <span class="c1"># 在前向星中加边</span>
    <span class="c1"># 如果要修改全局变量，要使用 global 来声明</span>
    <span class="k">global</span> <span class="n">cnt</span><span class="p">,</span> <span class="n">e</span><span class="p">,</span> <span class="n">h</span>
    <span class="c1"># 调试时的输出语句，多个变量使用元组</span>
    <span class="c1"># print("add_path(%d,%d,%d)" % (f,t,v))</span>
    <span class="n">cnt</span> <span class="o">+=</span> <span class="mi">1</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">nex</span> <span class="o">=</span> <span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">t</span> <span class="o">=</span> <span class="n">t</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">v</span> <span class="o">=</span> <span class="n">v</span>
    <span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span> <span class="o">=</span> <span class="n">cnt</span>
</code></pre></div></td></tr></tbody></table>

### Dijkstra 算法[](about:blank#dijkstra-%E7%AE%97%E6%B3%95 "Permanent link")

C++Python

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="kt">void</span><span class="w"> </span><span class="nf">dijkstra</span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">s</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">memset</span><span class="p">(</span><span class="n">dist</span><span class="p">,</span><span class="w"> </span><span class="mh">0x3f</span><span class="p">,</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">dist</span><span class="p">));</span>
<span class="w">  </span><span class="n">dist</span><span class="p">[</span><span class="n">s</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">q</span><span class="p">.</span><span class="n">push</span><span class="p">(</span><span class="n">make_pair</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">s</span><span class="p">));</span>
<span class="w">  </span><span class="k">while</span><span class="w"> </span><span class="p">(</span><span class="n">q</span><span class="p">.</span><span class="n">size</span><span class="p">())</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="n">pii</span><span class="w"> </span><span class="n">u</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">q</span><span class="p">.</span><span class="n">top</span><span class="p">();</span>
<span class="w">    </span><span class="n">q</span><span class="p">.</span><span class="n">pop</span><span class="p">();</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">u</span><span class="p">.</span><span class="n">first</span><span class="p">)</span><span class="w"> </span><span class="k">continue</span><span class="p">;</span>
<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">];</span><span class="w"> </span><span class="n">i</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">nex</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="k">const</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="o">&amp;</span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">w</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">v</span><span class="p">;</span>
<span class="w">      </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">w</span><span class="p">)</span><span class="w"> </span><span class="k">continue</span><span class="p">;</span>
<span class="w">      </span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">w</span><span class="p">;</span>
<span class="w">      </span><span class="n">q</span><span class="p">.</span><span class="n">push</span><span class="p">(</span><span class="n">make_pair</span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">],</span><span class="w"> </span><span class="n">v</span><span class="p">));</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">  </span><span class="p">}</span>
<span class="p">}</span>
</code></pre></div></td></tr></tbody></table>

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">def</span> <span class="nf">nextedgeid</span><span class="p">(</span><span class="n">u</span><span class="p">):</span>  <span class="c1"># 生成器，可以用在 for 循环里</span>
    <span class="n">i</span> <span class="o">=</span> <span class="n">h</span><span class="p">[</span><span class="n">u</span><span class="p">]</span>
    <span class="k">while</span> <span class="n">i</span><span class="p">:</span>
        <span class="k">yield</span> <span class="n">i</span>
        <span class="n">i</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">nex</span>


<span class="k">def</span> <span class="nf">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">):</span>
    <span class="n">dist</span><span class="p">[</span><span class="n">s</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">q</span><span class="o">.</span><span class="n">put</span><span class="p">((</span><span class="mi">0</span><span class="p">,</span> <span class="n">s</span><span class="p">))</span>
    <span class="k">while</span> <span class="ow">not</span> <span class="n">q</span><span class="o">.</span><span class="n">empty</span><span class="p">():</span>
        <span class="n">u</span> <span class="o">=</span> <span class="n">q</span><span class="o">.</span><span class="n">get</span><span class="p">()</span>  <span class="c1"># get 函数会顺便删除堆中对应的元素</span>
        <span class="k">if</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">&lt;</span> <span class="n">u</span><span class="p">[</span><span class="mi">0</span><span class="p">]:</span>
            <span class="k">continue</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">nextedgeid</span><span class="p">(</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
            <span class="n">v</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">t</span>
            <span class="n">w</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">v</span>
            <span class="k">if</span> <span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span> <span class="o">&lt;=</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">+</span> <span class="n">w</span><span class="p">:</span>
                <span class="k">continue</span>
            <span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span> <span class="o">=</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">+</span> <span class="n">w</span>
            <span class="n">q</span><span class="o">.</span><span class="n">put</span><span class="p">((</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">],</span> <span class="n">v</span><span class="p">))</span>
</code></pre></div></td></tr></tbody></table>

### 主函数[](about:blank#%E4%B8%BB%E5%87%BD%E6%95%B0 "Permanent link")

C++Python

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="kt">int</span><span class="w"> </span><span class="n">n</span><span class="p">,</span><span class="w"> </span><span class="n">m</span><span class="p">,</span><span class="w"> </span><span class="n">s</span><span class="p">;</span>

<span class="kt">int</span><span class="w"> </span><span class="nf">main</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">scanf</span><span class="p">(</span><span class="s">"%d%d%d"</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">n</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">m</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">s</span><span class="p">);</span>
<span class="w">  </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">m</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="n">w</span><span class="p">;</span>
<span class="w">    </span><span class="n">scanf</span><span class="p">(</span><span class="s">"%d%d%d"</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">w</span><span class="p">);</span>
<span class="w">    </span><span class="n">add_path</span><span class="p">(</span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="n">w</span><span class="p">);</span>
<span class="w">  </span><span class="p">}</span>
<span class="w">  </span><span class="n">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">);</span>
<span class="w">  </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">n</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="n">printf</span><span class="p">(</span><span class="s">"%d "</span><span class="p">,</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">i</span><span class="p">]);</span>
<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div></td></tr></tbody></table>

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="c1"># 一行读入多个整数。注意它会把整行都读进来</span>
    <span class="n">n</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="nb">int</span><span class="p">,</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">())</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">m</span><span class="p">):</span>
        <span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="nb">int</span><span class="p">,</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">())</span>
        <span class="n">add_path</span><span class="p">(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span>

    <span class="n">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
    
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">n</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
        <span class="nb">print</span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">end</span><span class="o">=</span><span class="s2">" "</span><span class="p">)</span>
    
    <span class="nb">print</span><span class="p">()</span>
</code></pre></div></td></tr></tbody></table>

### 完整代码[](about:blank#%E5%AE%8C%E6%95%B4%E4%BB%A3%E7%A0%81 "Permanent link")

C++Python

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span>
<span class="normal">26</span>
<span class="normal">27</span>
<span class="normal">28</span>
<span class="normal">29</span>
<span class="normal">30</span>
<span class="normal">31</span>
<span class="normal">32</span>
<span class="normal">33</span>
<span class="normal">34</span>
<span class="normal">35</span>
<span class="normal">36</span>
<span class="normal">37</span>
<span class="normal">38</span>
<span class="normal">39</span>
<span class="normal">40</span>
<span class="normal">41</span>
<span class="normal">42</span>
<span class="normal">43</span>
<span class="normal">44</span>
<span class="normal">45</span>
<span class="normal">46</span>
<span class="normal">47</span>
<span class="normal">48</span>
<span class="normal">49</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cstdio&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;cstring&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;queue&gt;</span>
<span class="cp">#include</span><span class="w"> </span><span class="cpf">&lt;vector&gt;</span>
<span class="k">using</span><span class="w"> </span><span class="k">namespace</span><span class="w"> </span><span class="nn">std</span><span class="p">;</span>
<span class="k">constexpr</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">N</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mf">1e5</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">5</span><span class="p">,</span><span class="w"> </span><span class="n">M</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mf">2e5</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="mi">5</span><span class="p">;</span>

<span class="k">struct</span><span class="w"> </span><span class="nc">qxx</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kt">int</span><span class="w"> </span><span class="n">nex</span><span class="p">,</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">;</span>
<span class="p">};</span>

<span class="n">qxx</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">M</span><span class="p">];</span>
<span class="kt">int</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">N</span><span class="p">],</span><span class="w"> </span><span class="n">cnt</span><span class="p">;</span>

<span class="kt">void</span><span class="w"> </span><span class="nf">add_path</span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">f</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="n">v</span><span class="p">)</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="o">++</span><span class="n">cnt</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">qxx</span><span class="p">{</span><span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">],</span><span class="w"> </span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">},</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">cnt</span><span class="p">;</span><span class="w"> </span><span class="p">}</span>

<span class="k">using</span><span class="w"> </span><span class="n">pii</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">pair</span><span class="o">&lt;</span><span class="kt">int</span><span class="p">,</span><span class="w"> </span><span class="kt">int</span><span class="o">&gt;</span><span class="p">;</span>
<span class="n">priority_queue</span><span class="o">&lt;</span><span class="n">pii</span><span class="p">,</span><span class="w"> </span><span class="n">vector</span><span class="o">&lt;</span><span class="n">pii</span><span class="o">&gt;</span><span class="p">,</span><span class="w"> </span><span class="n">greater</span><span class="o">&lt;</span><span class="n">pii</span><span class="o">&gt;&gt;</span><span class="w"> </span><span class="n">q</span><span class="p">;</span>
<span class="kt">int</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">N</span><span class="p">];</span>

<span class="kt">void</span><span class="w"> </span><span class="nf">dijkstra</span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">s</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">memset</span><span class="p">(</span><span class="n">dist</span><span class="p">,</span><span class="w"> </span><span class="mh">0x3f</span><span class="p">,</span><span class="w"> </span><span class="k">sizeof</span><span class="p">(</span><span class="n">dist</span><span class="p">));</span>
<span class="w">  </span><span class="n">dist</span><span class="p">[</span><span class="n">s</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">q</span><span class="p">.</span><span class="n">push</span><span class="p">(</span><span class="n">make_pair</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="w"> </span><span class="n">s</span><span class="p">));</span>
<span class="w">  </span><span class="k">while</span><span class="w"> </span><span class="p">(</span><span class="n">q</span><span class="p">.</span><span class="n">size</span><span class="p">())</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="n">pii</span><span class="w"> </span><span class="n">u</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">q</span><span class="p">.</span><span class="n">top</span><span class="p">();</span>
<span class="w">    </span><span class="n">q</span><span class="p">.</span><span class="n">pop</span><span class="p">();</span>
<span class="w">    </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">&lt;</span><span class="w"> </span><span class="n">u</span><span class="p">.</span><span class="n">first</span><span class="p">)</span><span class="w"> </span><span class="k">continue</span><span class="p">;</span>
<span class="w">    </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">h</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">];</span><span class="w"> </span><span class="n">i</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">nex</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="k">const</span><span class="w"> </span><span class="kt">int</span><span class="w"> </span><span class="o">&amp;</span><span class="n">v</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">t</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">w</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">].</span><span class="n">v</span><span class="p">;</span>
<span class="w">      </span><span class="k">if</span><span class="w"> </span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">w</span><span class="p">)</span><span class="w"> </span><span class="k">continue</span><span class="p">;</span>
<span class="w">      </span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">.</span><span class="n">second</span><span class="p">]</span><span class="w"> </span><span class="o">+</span><span class="w"> </span><span class="n">w</span><span class="p">;</span>
<span class="w">      </span><span class="n">q</span><span class="p">.</span><span class="n">push</span><span class="p">(</span><span class="n">make_pair</span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">],</span><span class="w"> </span><span class="n">v</span><span class="p">));</span>
<span class="w">    </span><span class="p">}</span>
<span class="w">  </span><span class="p">}</span>
<span class="p">}</span>

<span class="kt">int</span><span class="w"> </span><span class="n">n</span><span class="p">,</span><span class="w"> </span><span class="n">m</span><span class="p">,</span><span class="w"> </span><span class="n">s</span><span class="p">;</span>

<span class="kt">int</span><span class="w"> </span><span class="nf">main</span><span class="p">()</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="n">scanf</span><span class="p">(</span><span class="s">"%d%d%d"</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">n</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">m</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">s</span><span class="p">);</span>
<span class="w">  </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">m</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">    </span><span class="kt">int</span><span class="w"> </span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="n">w</span><span class="p">;</span>
<span class="w">    </span><span class="n">scanf</span><span class="p">(</span><span class="s">"%d%d%d"</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="o">&amp;</span><span class="n">w</span><span class="p">);</span>
<span class="w">    </span><span class="n">add_path</span><span class="p">(</span><span class="n">u</span><span class="p">,</span><span class="w"> </span><span class="n">v</span><span class="p">,</span><span class="w"> </span><span class="n">w</span><span class="p">);</span>
<span class="w">  </span><span class="p">}</span>
<span class="w">  </span><span class="n">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">);</span>
<span class="w">  </span><span class="k">for</span><span class="w"> </span><span class="p">(</span><span class="kt">int</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="mi">1</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="w"> </span><span class="o">&lt;=</span><span class="w"> </span><span class="n">n</span><span class="p">;</span><span class="w"> </span><span class="n">i</span><span class="o">++</span><span class="p">)</span><span class="w"> </span><span class="n">printf</span><span class="p">(</span><span class="s">"%d "</span><span class="p">,</span><span class="w"> </span><span class="n">dist</span><span class="p">[</span><span class="n">i</span><span class="p">]);</span>
<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="mi">0</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div></td></tr></tbody></table>

<table class="highlighttable"><tbody><tr><td class="linenos"><div class="linenodiv"><pre><span></span><span class="normal"> 1</span>
<span class="normal"> 2</span>
<span class="normal"> 3</span>
<span class="normal"> 4</span>
<span class="normal"> 5</span>
<span class="normal"> 6</span>
<span class="normal"> 7</span>
<span class="normal"> 8</span>
<span class="normal"> 9</span>
<span class="normal">10</span>
<span class="normal">11</span>
<span class="normal">12</span>
<span class="normal">13</span>
<span class="normal">14</span>
<span class="normal">15</span>
<span class="normal">16</span>
<span class="normal">17</span>
<span class="normal">18</span>
<span class="normal">19</span>
<span class="normal">20</span>
<span class="normal">21</span>
<span class="normal">22</span>
<span class="normal">23</span>
<span class="normal">24</span>
<span class="normal">25</span>
<span class="normal">26</span>
<span class="normal">27</span>
<span class="normal">28</span>
<span class="normal">29</span>
<span class="normal">30</span>
<span class="normal">31</span>
<span class="normal">32</span>
<span class="normal">33</span>
<span class="normal">34</span>
<span class="normal">35</span>
<span class="normal">36</span>
<span class="normal">37</span>
<span class="normal">38</span>
<span class="normal">39</span>
<span class="normal">40</span>
<span class="normal">41</span>
<span class="normal">42</span>
<span class="normal">43</span>
<span class="normal">44</span>
<span class="normal">45</span>
<span class="normal">46</span>
<span class="normal">47</span>
<span class="normal">48</span>
<span class="normal">49</span>
<span class="normal">50</span>
<span class="normal">51</span>
<span class="normal">52</span>
<span class="normal">53</span>
<span class="normal">54</span>
<span class="normal">55</span>
<span class="normal">56</span>
<span class="normal">57</span>
<span class="normal">58</span>
<span class="normal">59</span>
<span class="normal">60</span>
<span class="normal">61</span>
<span class="normal">62</span>
<span class="normal">63</span>
<span class="normal">64</span>
<span class="normal">65</span>
<span class="normal">66</span>
<span class="normal">67</span>
<span class="normal">68</span>
<span class="normal">69</span>
<span class="normal">70</span>
<span class="normal">71</span>
<span class="normal">72</span>
<span class="normal">73</span>
<span class="normal">74</span>
<span class="normal">75</span>
<span class="normal">76</span></pre></div></td><td class="code"><div><pre><span></span><code><span class="k">try</span><span class="p">:</span>  <span class="c1"># 引入优先队列模块</span>
    <span class="kn">import</span> <span class="nn">Queue</span> <span class="k">as</span> <span class="nn">pq</span>  <span class="c1"># python version &lt; 3.0</span>
<span class="k">except</span> <span class="ne">ImportError</span><span class="p">:</span>
    <span class="kn">import</span> <span class="nn">queue</span> <span class="k">as</span> <span class="nn">pq</span>  <span class="c1"># python3.*</span>

<span class="n">N</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mf">1e5</span> <span class="o">+</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">M</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mf">2e5</span> <span class="o">+</span> <span class="mi">5</span><span class="p">)</span>
<span class="n">INF</span> <span class="o">=</span> <span class="mh">0x3F3F3F3F</span>


<span class="k">class</span> <span class="nc">qxx</span><span class="p">:</span>  <span class="c1"># 前向星类（结构体）</span>
    <span class="k">def</span> <span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">nex</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">t</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">v</span> <span class="o">=</span> <span class="mi">0</span>


<span class="n">e</span> <span class="o">=</span> <span class="p">[</span><span class="n">qxx</span><span class="p">()</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">M</span><span class="p">)]</span>  <span class="c1"># 链表</span>
<span class="n">h</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>
<span class="n">cnt</span> <span class="o">=</span> <span class="mi">0</span>

<span class="n">dist</span> <span class="o">=</span> <span class="p">[</span><span class="n">INF</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">N</span><span class="p">)]</span>
<span class="n">q</span> <span class="o">=</span> <span class="n">pq</span><span class="o">.</span><span class="n">PriorityQueue</span><span class="p">()</span>  <span class="c1"># 定义优先队列，默认第一元小根堆</span>


<span class="k">def</span> <span class="nf">add_path</span><span class="p">(</span><span class="n">f</span><span class="p">,</span> <span class="n">t</span><span class="p">,</span> <span class="n">v</span><span class="p">):</span>  <span class="c1"># 在前向星中加边</span>
    <span class="c1"># 如果要修改全局变量，要使用 global 来声名</span>
    <span class="k">global</span> <span class="n">cnt</span><span class="p">,</span> <span class="n">e</span><span class="p">,</span> <span class="n">h</span>
    <span class="c1"># 调试时的输出语句，多个变量使用元组</span>
    <span class="c1"># print("add_path(%d,%d,%d)" % (f,t,v))</span>
    <span class="n">cnt</span> <span class="o">+=</span> <span class="mi">1</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">nex</span> <span class="o">=</span> <span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">t</span> <span class="o">=</span> <span class="n">t</span>
    <span class="n">e</span><span class="p">[</span><span class="n">cnt</span><span class="p">]</span><span class="o">.</span><span class="n">v</span> <span class="o">=</span> <span class="n">v</span>
    <span class="n">h</span><span class="p">[</span><span class="n">f</span><span class="p">]</span> <span class="o">=</span> <span class="n">cnt</span>


<span class="k">def</span> <span class="nf">nextedgeid</span><span class="p">(</span><span class="n">u</span><span class="p">):</span>  <span class="c1"># 生成器，可以用在 for 循环里</span>
    <span class="n">i</span> <span class="o">=</span> <span class="n">h</span><span class="p">[</span><span class="n">u</span><span class="p">]</span>
    <span class="k">while</span> <span class="n">i</span><span class="p">:</span>
        <span class="k">yield</span> <span class="n">i</span>
        <span class="n">i</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">nex</span>


<span class="k">def</span> <span class="nf">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">):</span>
    <span class="n">dist</span><span class="p">[</span><span class="n">s</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">q</span><span class="o">.</span><span class="n">put</span><span class="p">((</span><span class="mi">0</span><span class="p">,</span> <span class="n">s</span><span class="p">))</span>
    <span class="k">while</span> <span class="ow">not</span> <span class="n">q</span><span class="o">.</span><span class="n">empty</span><span class="p">():</span>
        <span class="n">u</span> <span class="o">=</span> <span class="n">q</span><span class="o">.</span><span class="n">get</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">&lt;</span> <span class="n">u</span><span class="p">[</span><span class="mi">0</span><span class="p">]:</span>
            <span class="k">continue</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">nextedgeid</span><span class="p">(</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
            <span class="n">v</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">t</span>
            <span class="n">w</span> <span class="o">=</span> <span class="n">e</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">v</span>
            <span class="k">if</span> <span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span> <span class="o">&lt;=</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">+</span> <span class="n">w</span><span class="p">:</span>
                <span class="k">continue</span>
            <span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">]</span> <span class="o">=</span> <span class="n">dist</span><span class="p">[</span><span class="n">u</span><span class="p">[</span><span class="mi">1</span><span class="p">]]</span> <span class="o">+</span> <span class="n">w</span>
            <span class="n">q</span><span class="o">.</span><span class="n">put</span><span class="p">((</span><span class="n">dist</span><span class="p">[</span><span class="n">v</span><span class="p">],</span> <span class="n">v</span><span class="p">))</span>


<span class="c1"># 如果你直接运行这个 Python 代码（不是模块调用什么的）就执行命令</span>
<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="c1"># 一行读入多个整数。注意它会把整行都读进来</span>
    <span class="n">n</span><span class="p">,</span> <span class="n">m</span><span class="p">,</span> <span class="n">s</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="nb">int</span><span class="p">,</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">())</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">m</span><span class="p">):</span>
        <span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span> <span class="o">=</span> <span class="nb">map</span><span class="p">(</span><span class="nb">int</span><span class="p">,</span> <span class="nb">input</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">())</span>
        <span class="n">add_path</span><span class="p">(</span><span class="n">u</span><span class="p">,</span> <span class="n">v</span><span class="p">,</span> <span class="n">w</span><span class="p">)</span>

    <span class="n">dijkstra</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
    
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">n</span> <span class="o">+</span> <span class="mi">1</span><span class="p">):</span>
        <span class="c1"># 两种输出语法都是可以用的</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="si">{}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">dist</span><span class="p">[</span><span class="n">i</span><span class="p">]),</span> <span class="n">end</span><span class="o">=</span><span class="s2">" "</span><span class="p">)</span>
        <span class="c1"># print("%d" % dist[i],end=' ')</span>
    
    <span class="nb">print</span><span class="p">()</span>  <span class="c1"># 结尾换行</span>
</code></pre></div></td></tr></tbody></table>

参考文档[](about:blank#%E5%8F%82%E8%80%83%E6%96%87%E6%A1%A3 "Permanent link")
-------------------------------------------------------------------------

1.  [Python Documentation](https://www.python.org/doc/)
2.  [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/)
3.  [Learn Python3 In Y Minutes](https://learnxinyminutes.com/docs/python3/)
4.  [Real Python Tutorials](https://realpython.com/)
5.  [廖雪峰的 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400/)
6.  [GeeksforGeeks: Python Tutorials](https://www.geeksforgeeks.org/python-programming-language/)

参考资料和注释[](about:blank#%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99%E5%92%8C%E6%B3%A8%E9%87%8A "Permanent link")
-------------------------------------------------------------------------------------------------------

* * *

1.  [2\. Python 解释器—Python 3 文档](https://docs.python.org/zh-cn/3/tutorial/interpreter.html#id1) [↩](about:blank#fnref:ref1 "Jump back to footnote 1 in the text")
    
2.  [Unicode 指南—Python 3 文档](https://docs.python.org/zh-cn/3/howto/unicode.html#the-string-type) [↩](about:blank#fnref:ref2 "Jump back to footnote 2 in the text")
    

* * *

> 本页面最近更新：2025/2/2 13:28:46，[更新历史](https://github.com/OI-wiki/OI-wiki/commits/master/docs/lang/python.md)  
> 发现错误？想一起完善？ [在 GitHub 上编辑此页！](https://oi-wiki.org/edit-landing/?ref=/lang/python.md "edit.link.title")  
> 本页面贡献者：[cmpute](https://github.com/cmpute), [Henry-ZHR](https://github.com/Henry-ZHR), [ranwen](https://github.com/ranwen), [abc1763613206](https://github.com/abc1763613206), [billchenchina](https://github.com/billchenchina), [chinggg](https://github.com/chinggg), [ChungZH](https://github.com/ChungZH), [CoelacanthusHex](https://github.com/CoelacanthusHex), [countercurrent-time](https://github.com/countercurrent-time), [Early0v0](https://github.com/Early0v0), [Enter-tainer](https://github.com/Enter-tainer), [F1shAndCat](https://github.com/F1shAndCat), [Great-designer](https://github.com/Great-designer), [hensier](https://github.com/hensier), [HeRaNO](https://github.com/HeRaNO), [Hszzzx](https://github.com/Hszzzx), [imba-tjd](https://github.com/imba-tjd), [Ir1d](https://github.com/Ir1d), [jiangmuran](https://github.com/jiangmuran), [ksyx](https://github.com/ksyx), [lingxier](https://github.com/lingxier), [LovelyBuggies](https://github.com/LovelyBuggies), [Marcythm](https://github.com/Marcythm), [Mooos-MoSheng](https://github.com/Mooos-MoSheng), [NachtgeistW](https://github.com/NachtgeistW), [ouuan](https://github.com/ouuan), [Rottenwooood](https://github.com/Rottenwooood), [shawlleyw](https://github.com/shawlleyw), [shuzhouliu](https://github.com/shuzhouliu), [sshwy](https://github.com/sshwy), [SukkaW](https://github.com/SukkaW), [Tiphereth-A](https://github.com/Tiphereth-A), [tLLWtG](https://github.com/tLLWtG), [wineee](https://github.com/wineee), [wxh06](https://github.com/wxh06), [Xeonacid](https://github.com/Xeonacid), [yusancky](https://github.com/yusancky), [zyouxam](https://github.com/zyouxam), [zzjjbb](https://github.com/zzjjbb), [Dong Tsing-hsuen](mailto:68433824+alissa42@users.noreply.github.com), [Dong Tsing-hsuen](https://github.com/Dong%20Tsing-hsuen), [mgt](mailto:i@margatroid.xyz), [mgt](https://github.com/mgt), [Suyun514](mailto:suyun514@qq.com), [Suyun514](https://github.com/Suyun514)  
> 本页面的全部内容在 **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 和 [SATA](https://github.com/zTrix/sata-license)** 协议之条款下提供，附加条款亦可能应用

var comments=document.createElement("script"),commentsTheme="slate"===document.body.dataset.mdColorScheme?"dark":"light";Object.entries({async:!0,src:"https://giscus.app/client.js",crossOrigin:"anonymous","data-repo":"OI-wiki/gitment","data-repo-id":"MDEwOlJlcG9zaXRvcnkxNDQzODg5NjU=","data-category":"评论","data-category-id":"DIC\_kwDOCJszZc4CS54y","data-mapping":"specific","data-term":"Python 速成","data-strict":"1","data-reactions-enabled":"1","data-emit-metadata":"0","data-input-position":"top","data-theme":commentsTheme,"data-lang":"zh-CN","data-loading":"lazy"}).forEach((e=>comments.setAttribute(e\[0\],e\[1\]))),document.getElementById("\_\_comments\_script").replaceWith(comments)

  

本文转自 [https://oi-wiki.org/lang/python/](https://oi-wiki.org/lang/python/)，如有侵权，请联系删除。