from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 测试数据
x = np.array([1, 2, 3, 4, 5])

# 创建 PolynomialFeatures 对象，指定最高次为 2
poly_features = PolynomialFeatures(degree=2)

# 使用 fit_transform 进行特征扩展
x_poly = poly_features.fit_transform(x.reshape(-1, 1))

# 打印扩展后的特征数据
print(x_poly)