from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix

from utils import PlotConfusionMatrix       # 绘制混淆矩阵的工具类
from utils import PlotPrecisionRecallCurve  # 绘制PR曲线的工具类
from utils import PlotRocCurve              # 绘制Roc曲线工具类

def logistic_regression_confusion_matrix():
 # 加载乳腺癌数据集
 cancer = load_breast_cancer()
 X = cancer.data
 y = cancer.target

 # 将数据集分为训练集和测试集
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # 创建逻辑回归模型
 model = LogisticRegression()

 # 在训练集上训练模型
 model.fit(X_train, y_train)

 # 在测试集上进行预测
 y_pred = model.predict(X_test)
 # 计算y_score：是一个含有多类别概率的向量
 y_score = model.decision_function(X_test)

 # 在混淆矩阵中进行分析
 result = confusion_matrix(y_test,y_pred)
 print(result)

 # 绘制混淆矩阵
 PlotConfusionMatrix.plot_confusion_matrix(y_test,y_pred)

 # 绘制PRC曲线
 PlotPrecisionRecallCurve.plot_precision_recall_curve(y_test,y_score)

 # 绘制ROC曲线并计算AUC值
 PlotRocCurve.plot_roc_curve(y_test,y_score)

logistic_regression_confusion_matrix()