# 定义训练数据的输入值和目标输出值
x_data = [1.0, 2.0, 3.0]  # 输入数据列表
y_data = [2.0, 4.0, 6.0]  # 目标输出数据列表

w = 1.0  # 初始化权重参数 w，初始值为 1.0


# 定义前向传播函数，用于计算模型预测值
def forward(x):
    return x * w  # 线性模型：y = w * x


# 定义损失函数，用于计算模型预测值与真实值之间的误差
def cost(xs, ys):
    cost = 0  # 初始化损失值为 0
    for x, y in zip(xs, ys):  # 遍历输入数据和目标输出数据
        y_pred = forward(x)  # 计算预测值
        cost += (y_pred - y) ** 2  # 计算平方误差并累加
    return cost / len(xs)  # 返回平均损失值


# 定义梯度计算函数，用于计算损失函数对权重 w 的梯度
def gradient(xs, ys):
    grad = 0  # 初始化梯度为 0
    for x, y in zip(xs, ys):  # 遍历输入数据和目标输出数据
        grad += 2 * x * (x * w - y)  # 计算梯度并累加（基于平方误差公式求导）
    return grad / len(xs)  # 返回平均梯度值


# 打印训练前的预测值
print("Predict (before training)", 4, forward(4))  # 使用初始权重 w 预测输入 4 的输出

# 开始训练模型
for epoch in range(100):  # 迭代 100 次进行训练
    cost_val = cost(x_data, y_data)  # 计算当前权重下的损失值
    grad_val = gradient(x_data, y_data)  # 计算当前权重下的梯度值
    w -= 0.01 * grad_val  # 根据梯度更新权重参数 w
    print("Epoch:", epoch, "w=", w, "loss=", cost_val)

# 打印训练后的预测值
print("Predict (after training)", 4, forward(4))  # 使用训练后的权重 w 预测输入 4 的输出
