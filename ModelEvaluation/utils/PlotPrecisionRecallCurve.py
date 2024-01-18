import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(y_true, y_scores):

    # 计算精度和召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # 绘制 PRC 曲线
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# 示例数据
# y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
# y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])

# plot_precision_recall_curve(y_true, y_scores)