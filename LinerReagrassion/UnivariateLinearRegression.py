from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 生成示例数据
x = np.array([2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 4, 5])

# 将数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 在训练集上拟合模型
model.fit(x_train.reshape(-1, 1), y_train)

# 在测试集上进行预测
y_pred = model.predict(x_test.reshape(-1, 1))

# 计算预测误差
error = mean_squared_error(y_test, y_pred)
print("预测误差：", error)