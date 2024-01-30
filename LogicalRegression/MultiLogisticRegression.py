from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def multinomial_logistic_regression(X, y):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建逻辑回归模型并进行训练，multi_class='multinomial'配置为多分类模型
    model = LogisticRegression(multi_class='multinomial', random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target


# 使用多分类逻辑回归进行预测
multinomial_logistic_regression(X, y)