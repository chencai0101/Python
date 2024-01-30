from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def logistic_regression(X,y):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression() # 创建逻辑回归模型
    model.fit(X_train, y_train) # 在训练集上训练模型
    y_pred = model.predict(X_test) # 在测试集上进行预测
    accuracy = accuracy_score(y_test, y_pred) # 计算准确率
    return accuracy,model

# 读取 CSV 数据集
df = pd.read_csv('Resource/data/iris_binary.csv')

# 提取 sepal_length，sepal_width，class 列的数据，并转化为numpy数组
sepal_length = df['sepal_length'].values
sepal_width = df['sepal_width'].values
y = df['class'].values

# 合并特征列
X = np.concatenate((sepal_length[:,np.newaxis],sepal_width[:,np.newaxis]),axis=1)

# 调用函数进行逻辑回归
accuracy,model = logistic_regression(X,y)

# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

plt.xlabel('sepal_length_data') # 设置坐标轴标签
plt.ylabel('sepal_width')
plt.title('iris_binary')

# 绘制决策边界
slope=-model.coef_[0][0]/model.coef_[0][1]              # 斜率
intercept=-model.intercept_[0]/model.coef_[0][1]        # 截距
plt.plot(sepal_length, [x*slope+intercept for x in sepal_length])  

plt.show()  # 显示图形