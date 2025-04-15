import torch

# 输入数据 x_data 和目标数据 y_data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重 w 为一个张量，并设置 requires_grad=True，表示需要计算梯度
w = torch.tensor([1.0])
w.requires_grad = True


def forward(x):
    """
    前向计算函数，计算预测值 y_pred = x * w。
    - w 是一个 tensor，因此支持自动求导。
    - x 是标量，计算时会自动转换为 tensor。
    """
    return x * w


def loss(x, y):
    """
    损失函数，计算预测值与实际值之间的均方误差 (MSE)。
    - y_pred: 通过 forward 计算的预测值。
    - y: 实际值。
    返回值是一个张量，包含损失值。
    """
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 打印训练前的预测值
print("predict (before training)", 4, forward(4).item())

# 开始训练，训练 100 个 epoch
for epoch in range(100):
    # 遍历每个数据点 (x, y)
    for x, y in zip(x_data, y_data):
        # 计算损失值
        l = loss(x, y)
        # 进行反向传播，计算梯度
        l.backward()
        print("\tgrad: ", x, y, w.grad.item())
        # 更新权重 w，使用梯度下降法更新
        # w.grad 也是一个张量，因此通过 .data 获取其值，避免记录在计算图中
        w.data = w.data - 0.01 * w.grad.data
        # 梯度清零，准备下一次迭代
        w.grad.data.zero_()
    # 打印每个 epoch 的损失值
    print("progress:", epoch, l.item())

# 打印训练后的预测值
print("predict (after training)", 4, forward(4).item())
