from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

def PolynomialRegression(degree): 
    # 创建多项式特征：degree指定次数
    polynomial_features = PolynomialFeatures(degree=degree)

    # 创建线性回归模型
    linear_regression = LinearRegression()

    # 创建管道：将线性回归模型与多项式特征组合
    pipeline = Pipeline([
        ('polynomial_features', polynomial_features),
        ('linear_regression', linear_regression)
    ])

    return pipeline

# 生成数据
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 0.1, 100).reshape(-1, 1)  # 加上这个随机值是为了让数据出现波动

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 创建多项式回归模型与多项式特征生成器连在一起，生成一个整体的模型。
model = PolynomialRegression(degree=2)

# 拟合模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 打印预测结果
print("Test MSE:", np.mean((y_test - y_pred)**2))

# 画图展示拟合效果
plt.scatter(X, y)
plt.plot(X,model.predict(X),"r-")   # 绘制为红色实线，参数1：一定是一个x的等差序列；参数2则是对下x序列进行计算
                                    # 这里传x_test为何不行呢？因为划分数据集后x_test乱序了……        
# 添加标题和轴标签
plt.title('PolynomialRegression')
plt.xlabel('x')
plt.ylabel('y')
# 显示图形
plt.show()