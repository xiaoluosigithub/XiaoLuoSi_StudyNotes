# 导入 PyTorch 的神经网络模块和功能模块
import torch.nn
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 定义逻辑回归模型类，继承自 torch.nn.Module
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        # 调用父类构造函数进行初始化
        super(LogisticRegressionModel, self).__init__()
        # 定义一个线性层，输入维度为 1，输出维度为 1
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 在前向传播中，将输入通过线性层后应用 sigmoid 激活函数
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


# 实例化逻辑回归模型
model = LogisticRegressionModel()

# 定义损失函数为二元交叉熵损失（BCELoss），注意 size_average 参数已被弃用，应使用 reduction 参数
criterion = torch.nn.BCELoss(reduction='sum')
# 定义优化器为随机梯度下降（SGD），学习率为 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 开始训练循环，共训练 1000 轮
for epoch in range(1000):
    # 前向传播：计算预测值 y_pred
    y_pred = model(x_data)
    # 计算损失值
    loss = criterion(y_pred, y_data)
    # 打印当前轮次和损失值
    print(epoch, loss.item())

    # 将优化器中的梯度清零，避免累加
    optimizer.zero_grad()
    # 反向传播：计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

print('w = ', model.linear.weight.item())
