from sklearn.pipeline import Pipeline                   # 管道
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.preprocessing import PolynomialFeatures    # 多项式特征
from sklearn.linear_model import LinearRegression       # 线性回归模型
from sklearn.model_selection import train_test_split    # 划分数据
import numpy as np

# 实例化相关模块
poly_features = PolynomialFeatures
standardScaler = StandardScaler
linearRegression = LinearRegression

# 配置流水线
pipeline = Pipeline([
    ("poly_features", poly_features(degree=2)),
    ("standardScaler", standardScaler()),
    ("linearRegression", linearRegression())
])

# 准备数据
X = np.array([[2, 3], [4, 5], [2, 7], [4, 5], [6, 9]]) 
y = np.array([1, 2, 3, 4, 5])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 流水线编排好了直接一键训练！
pipeline.fit(x_train, y_train)
predict = pipeline.predict(x_test)

print(predict)