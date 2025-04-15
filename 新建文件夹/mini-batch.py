# 数据集定义
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始权重
w = 1.0


# 前向传播函数
def forward(x):
    return x * w


# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 计算梯度
def gradient(x, y):
    return 2 * x * (x * w - y)


# 打印预测值（训练前）
print("预测（训练前）", 4, forward(4))

# 训练过程
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print("\t梯度: ", x, y, grad)
        l = loss(x, y)
    print("进度:", epoch, "系数w=", w, "目标值和真实值偏差=", l)

# 打印预测值（训练后）
print("预测（训练后）", 4, forward(4))
