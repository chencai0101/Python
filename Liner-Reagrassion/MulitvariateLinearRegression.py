from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import numpy as np

# 生成示例数据
x = np.array([[2, 3],[ 4, 5],[2, 7],[4, 5],[6, 9]])
y = np.array([1, 2, 3, 4, 5])

# 将数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 在训练集上拟合模型
model.fit(x_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(x_test)

# 获取模型参数
print("斜率: ", model.coef_)
print("截距: ", model.intercept_)

# 计算预测误差
error = mean_squared_error(y_test, y_pred)
print("预测误差：", error)

# 绘制三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_scatter = np.array([2, 4, 2, 4, 6])
y_scatter = np.array([3, 5, 7, 5, 9])
z_scatter = np.array([1, 2, 3, 4, 5])

# 生成超平面上的点：返回两个等差数列s
x_surface = np.linspace(-10, 10, 100)                             # x_surface与y_surface实际上指定了超平面网格的密度
y_surface = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_surface, y_surface)                          # 使用np.meshgrid函数将这两个等差数列组合成一个二维网格，
Z = model.coef_[0] * X + model.coef_[1] * Y + model.intercept_    # 计算出这些网格点上的z轴高度

ax.scatter(x_scatter, y_scatter, z_scatter, c=z_scatter, cmap='viridis')  # 使用 z 值作为颜色映射，根据高度的变化而变化。
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()