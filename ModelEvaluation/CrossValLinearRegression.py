from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

def cross_val_linear_regression(X, y, cv=3):
    # 创建线性回归模型
    model = LinearRegression()
    # 使用 cross_val_score 进行交叉验证
    scores = cross_val_score(model, X, y, cv=cv)
    # 返回平均得分
    return scores, scores.mean()

# 示例用法
X = np.array([[2, 3],[ 4, 5],[2, 7],[4, 5],[6, 9],[4, 7],[ 6, 5],[6, 8],[2, 5],[9, 9]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cv_score,cv_score_mean = cross_val_linear_regression(X, y)

print("每一折得分:",cv_score)
print("交叉验证平均得分:", cv_score_mean)