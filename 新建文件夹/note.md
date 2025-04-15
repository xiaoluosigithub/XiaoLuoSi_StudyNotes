## Mini-batch

在深度学习中，我们训练模型的目标是通过一组样本数据来更新模型的参数。这个过程中涉及到一个叫 **梯度下降（Gradient Descent）** 的优化方法。而 **mini-batch** 就是梯度下降中的一种 **数据处理方式**。

通俗地说：

> **mini-batch 就是“每次训练只用部分数据”来更新模型，而不是一次用全部或一条数据。**

假设你有一个数据集，共有 **10000 条样本**。你设置 mini-batch 大小为 100。那么：

1. 你会将整个数据集**切成 100 个小批次**，每个批次 100 条样本。
2. 模型每次从一个 mini-batch 中取出数据，计算损失函数和梯度，更新模型参数。
3. 一轮训练结束，称为 **一个 epoch**。
4. 下一轮重新分组（可打乱顺序），继续训练。

| 优点       | 说明                                            |
| ---------- | ----------------------------------------------- |
| 💨 更快     | 相比整批数据，mini-batch 每次处理数据少，迭代快 |
| 📉 稳定性好 | 相比一条数据（SGD），平均多个样本的梯度更稳定   |
| 🧠 更好泛化 | 随机选取小批量，有助于模型学习更一般化的特征    |
| 🧮 硬件友好 | 特别适合 GPU 并行处理                           |



## 系数 = 系数 - 学习率 * grad

`grad` 是导数，表示 **当前参数对损失函数的影响方向和大小**

假设我们的损失函数是：

$$
Loss(w) = (w * x - y)^2
$$
我们对 `w` 求导数，得到：

$$
grad=dLoss/dw=2x(w∗x−y)
$$
这个导数告诉我们：

> 如果我们把 `w` **稍微加一点点**，损失函数是会上升还是下降。

✅ 如果 `grad > 0`

说明 `w` 太大了，应该往**小方向**调整。
 ➡ 所以我们要 `w = w - α * grad`

✅ 如果 `grad < 0`

说明 `w` 太小了，应该往**大方向**调整。
 ➡ 还是用同样的公式：`w = w - α * grad`，这时减负数，相当于加！



## 大型神经网络是怎么“自己求导数、更新参数”的呢？

**自动求导**（Autograd）这是 PyTorch、TensorFlow 等深度学习框架的“杀手级功能”。

### 🔄 什么是自动求导？

自动求导的本质就是：**用程序来自动计算每一个参数的梯度**。

它的核心思想叫做：

> **反向传播（backpropagation） + 链式法则（chain rule）**



~~~python
import torch

# 定义数据和参数（带 grad）
x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)

# 定义前向传播
y = x * w
loss = (y - 6)**2  # 目标值是6

# 反向传播
loss.backward()

# 查看梯度
print(w.grad)  # 输出梯度：2 * x * (x*w - y) = 2 * 2 * (6 - 6) = 0
~~~



~~~python
x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
~~~

一旦你用 `torch.tensor(...)` 创建变量，**它就是 PyTorch 的张量对象 `torch.Tensor`**，而不是 Python 内建的 `float` 或 `int` 类型。

只有 `torch.Tensor` 类型，并且设置了 `requires_grad=True` 的变量，才能被 PyTorch **跟踪和记录操作历史**，从而实现自动求导！

### 自动反向传播

~~~python
loss.backward()
~~~

这是最重要的一行：

- 告诉 PyTorch：“现在请你**从 loss 开始，反向传播计算每个参数的梯度**。”
- PyTorch 会自动使用“链式法则”（Chain Rule）来计算所有带 `requires_grad=True` 的变量的 `.grad` 值。



小补充：`requires_grad` 的作用非常重要

| 参数                          | 含义                                            |
| ----------------------------- | ----------------------------------------------- |
| `requires_grad=False`（默认） | 不会记录梯度，也不会参与训练                    |
| `requires_grad=True`          | 表示这是**要训练的参数**，需要 PyTorch 自动求导 |

### 问题一

**如果我有两个结果（比如两个 loss），分别调用 `.backward()`，那 `x.grad` 和 `w.grad` 到底会存储哪一个的梯度**

PyTorch 的 `.grad` **默认只保存**：

> **上一次 `.backward()` 计算出来的梯度值，会覆盖之前的！**

如果你想**同时保留多个 `.backward()` 的影响**，你必须手动控制 —— **比如用 `retain_graph=True` 或者手动累加。**

### 问题二

你有一个 **计算链**：

```nginx
w ─┐
   ├──> y = x * w ──> z = y - 6 ──> loss = z²
x ─┘
```

然后你先做：

```python
z.backward()
```

再做：

```python
loss.backward()
```

然后你问：

> 现在 `w.grad` 存的到底是谁的梯度？是 `z` 对 `w` 的导数，还是 `loss` 对 `w` 的导数？

------

🎯 标准答案：

▶ `w.grad` 会存储的是 **两个 `.backward()` 计算的“累加值”**！

也就是说：

> **如果你没有在中间清除 `.grad`，它就会自动累加所有 `.backward()` 的梯度。**

❗ 所以你上面的情况中：

```python
z.backward()        # 会计算 dz/dw，并加到 w.grad 上
loss.backward()     # 会计算 dloss/dw，并加到 w.grad 上
```

最终：

```python
w.grad = dz/dw + dloss/dw
```

### 问题三

为什么是将两个求导结果相加呢？为什么要这样设计呢？

梯度加法的本质

🚀 1. **梯度** 是一个 **局部的变化率**

当我们计算损失函数（loss）对一个变量（例如 `w`）的梯度时，梯度表达了如果改变 `w` 的值，**损失函数**的变化量。

假设你有两个损失函数 `loss1` 和 `loss2`，它们依赖于相同的变量 `w`，并且在训练过程中你希望通过两个损失函数的影响来更新 `w`，那么**最终的梯度**应该是两个损失函数对 `w` 的 **梯度之和**。这是因为每一个损失函数 **独立地影响 `w`**，这两者的影响是可以叠加的。

🚧 2. **链式法则的支持**

根据链式法则，梯度的传播是逐步的。在 **多次 `.backward()`** 的情况下，每个损失函数的导数都代表了该损失函数相对于 `w` 的变化。你可以把这个过程理解为 **两个函数的梯度传递的合成**，因此最终的梯度是它们的加和。

例如：

- `loss1.backward()` 会计算 `∂loss1/∂w`
- `loss2.backward()` 会计算 `∂loss2/∂w`

如果你在同一个计算图中依赖于 `w` 的两个独立损失函数，那么总的梯度就应该是：
$$
\frac{\partial loss_1}{\partial w} + \frac{\partial loss_2}{\partial w}
$$
 **简化的类比：**

假设你有两个不同的错误来源，它们都影响了你做事的最终结果。你要采取的行动是同时修正这两个错误，而不是单独修正其中一个。通过加起来，你能一次性修正两个问题！

### [矩阵计算公式](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)



## 激活函数

![PixPin_2025-04-15_20-01-42](.\images\PixPin_2025-04-15_20-01-42.png)

我们发现不断进行线性变换，化简，最终，不管有多少层，都会化简为一层，因此我们在每一层后面都加上一层非线性的函数，来防止这种化简

- **两层神经网络的最终表达式化简后仍然是线性模型**，其本质上与单层神经网络没有区别。
- 如果没有引入非线性函数，网络的表达能力会受到限制，无法拟合复杂的非线性关系。

## pytorch tensor

![image-20250415201743003](images\image-20250415201743003.png)



## Pytorch_fashion

准备数据集

~~~python
x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([2.0, 4.0, 6.0])
~~~

构造模型

![image-20250415205616662](images\image-20250415205616662.png)

~~~python
class LinearModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的构造
        super(LinearModel, self).__init__()
        # 构造一个对象linear 包含权重和偏置这两个参数
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
~~~

为什么没有backward函数？

因为model构造出的对象它会根据你的计算图自动的实现backward的过程



构造损失函数和优化器

~~~python
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
~~~

训练周期

~~~python
for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # 进行更新，根据参数和预先设置的学习率进行自动更新
~~~



### pytorch广播机制

在 PyTorch 中，广播机制（Broadcasting）是一种方便的功能，允许在形状不同的张量之间进行算术操作（如加法、乘法等），无需显式地调整维度。广播会自动扩展较小的张量以匹配较大的张量形状，从而实现操作。

广播机制基于以下两条规则：

1. **从后向前匹配维度**：
   - 从两个张量的最后一个维度开始，逐个比较两个张量的维度。
   - 如果维度相同或其中一个维度为 1，则可以进行广播。
2. **缺失的维度视为 1**：
   - 如果一个张量的维度比另一个张量少，则在前面补充维度 1 进行匹配。

通过广播扩展张量时，不会真实地复制数据，而是通过在计算时“虚拟地”扩展形状来完成操作。这种方式既节省内存，又提高了效率。

示例 1：标量和张量的操作

~~~python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor(2)
result = a + b  # 广播：b 被扩展为 [2, 2, 2]
print(result)  # 输出 [3, 4, 5]
~~~

示例 2：二维张量和一维张量的操作

~~~python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([1, 2, 3])
result = a + b  # 广播：b 被扩展为 [[1, 2, 3], [1, 2, 3]]
print(result)
~~~

示例 3：不同形状的多维张量操作

~~~python
a = torch.tensor([[[1, 2, 3]], [[4, 5, 6]]])  # 形状 (2, 1, 3)
b = torch.tensor([[1, 2, 3]])                 # 形状 (1, 3)
result = a + b  # 广播：b 被扩展为 (2, 1, 3)
print(result)
~~~

广播机制的限制

- 如果两个张量在某个维度上不相等且其中一个维度也不是 1，则会报错。
- 广播机制要求形状能按照上述规则兼容，否则无法完成操作。