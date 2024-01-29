import pandas as pd

# 读取 CSV 数据集
df = pd.read_csv('Resource/data/iris_binary.csv')

# 提取 sepal_length，sepal_width，class 列的数据
sepal_length_data = df['sepal_length']
sepal_width = df['sepal_width']
y = df['class']

# 合并特征列，并将其处理为numpy数组
X = pd.concat([sepal_length_data, sepal_width]).values.reshape(-1,2)
y = y.values

print(X)
print(y)
