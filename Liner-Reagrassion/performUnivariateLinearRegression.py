# 机器学习经典三件套
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# 导入刚写的线性回归模块
from LinearRegrassion import LinearRegrassion

# 读取数据
data = pd.read_csv('Liner-Reagrassion\data\world-happiness-report-2017.csv')

# 划分数据集
train_data = data.sample(frac=0.8)       # 选择80%的数据作为训练集
test_data = data.drop(train_data.index)  # 剩余的部分作为测试集

# 指定输入特征与输出特征
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 由于没有全部参加训练，所以指定参与训练与测试的字段; data.value 转化为ndarray（Numpy数组从而支持更多Numpy操作）
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# 画一个散点图看看结果
plt.scatter(x_train,y_train,label ='Train data')
plt.scatter(x_test,y_test,label ='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("liner regression")
# plt.show()

# 开始训练
num_iteration = 1000                                    # 指定训练次数
learning_rate = 0.01                                    # 指定学习率
linear_regrassion = LinearRegrassion(x_train,y_train)   # 实例化线性回归模型
(theta,cost_history) = linear_regrassion.train(learning_rate,num_iteration)

print(cost_history)

