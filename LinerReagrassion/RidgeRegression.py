from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def ridge_regression(X, y, lambda_):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建岭回归模型
    ridge = Ridge(alpha=lambda_)
    ridge.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = ridge.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    return mse

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 调整 lambda 参数进行交叉验证
lambda_values = [0, 0.01, 0.1, 1, 10, 100]
errors = []
for lambda_ in lambda_values:
    error = ridge_regression(X, y, lambda_)
    errors.append(error)

# 打印结果
print("正则化力度与均方误差的关系：")
for i in range(len(lambda_values)):
    print(f"Lambda: {lambda_values[i]}, MSE: {errors[i]}")

# 绘制折线图展示    
plt.plot(lambda_values,errors,"rs-")
plt.show()