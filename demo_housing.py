import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 数据预处理
data = pd.read_csv("data/HousingData.csv", sep=",")

# 处理缺失值（用均值填充）
# data = data.fillna(data.mean())

# 数据分割
x_train = data.iloc[:496, :-1].values
y_train = data.iloc[:496, -1].values.reshape(-1, 1)  # 转为二维数组方便标准化
x_test = data.iloc[496:, :-1].values
y_test = data.iloc[496:, -1].values.reshape(-1, 1)

# 数据标准化
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 转换为PyTorch张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 网络定义
class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, output_size)

        # 初始化权重
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


net = Net(input_size=13, output_size=1)

# 损失函数和优化器
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# 训练循环
for epoch in range(1000):
    # 前向传播
    y_pred = net(x_train_tensor)

    # 计算损失（移除了0.1缩放因子）
    l = loss(y_pred, y_train_tensor)

    # 反向传播
    optimizer.zero_grad()
    l.backward()

    # 梯度裁剪防止爆炸
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {l.item():.4f}")

# 测试结果
with torch.no_grad():
    y_pred_test = net(x_test_tensor)
    test_loss = loss(y_pred_test, y_test_tensor)
    print(f"\nTest Loss: {test_loss.item():.4f}")

    # 反标准化输出结果
    y_pred_actual = scaler_y.inverse_transform(y_pred_test.numpy())
    y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())
    print("\nPredicted vs Actual values:")
    for pred, actual in zip(y_pred_actual[:5], y_test_actual[:5]):
        print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]:.2f}")
