# 机器学习经典三件套
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# 导入刚写的线性回归模块
from LinearRegrassionModel import LinearRegrassion

# 读取数据
data = pd.read_csv('Liner-Reagrassion\data\world-happiness-report-2017.csv')

# 划分数据集
train_data = data.sample(frac=0.8)       # 选择80%的数据作为训练集
test_data = data.drop(train_data.index)  # 剩余的部分作为测试集

# 指定输入特征与输出特征
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# 由于没有全部参加训练，所以指定参与训练与测试的字段; data.value 转化为Numpy数组从而支持算法计算
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values
x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# 画一个散点图看看数据集的分布情况
plt.scatter(x_train,y_train,label ='Train data')
plt.scatter(x_test,y_test,label ='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("liner regression")
plt.show()

# 开始训练
num_iteration = 1000                                    # 指定训练次数
learning_rate = 0.01                                    # 指定学习率
linear_regrassion = LinearRegrassion(x_train,y_train)   # 实例化线性回归模型
(theta,cost_history) = linear_regrassion.train(learning_rate,num_iteration)

# 绘制出迭代次数与代价的关系图
plt.plot(range(num_iteration),cost_history)
plt.xlabel("num_literation")
plt.ylabel("cost")
plt.title("the relationship between iteration count and cost")
plt.show()

# 绘制出训练好的模型与原始数据散点图之间的关系
num_prediction = 100
x_coordinate_sequence = np.linspace(x_train.min(),x_train.max(),num_prediction).reshape(num_prediction,1) # 在训练集x的最大最小值之间划分出一个等差数列，并将其修改为 100 行 1 列的形状 
y_coordinate_sequence = linear_regrassion.predict(x_coordinate_sequence)

plt.scatter(x_train,y_train,label ='Train data')
plt.scatter(x_test,y_test,label ='Test data')
plt.plot(x_coordinate_sequence,y_coordinate_sequence,"r")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("liner regression")
plt.show()

