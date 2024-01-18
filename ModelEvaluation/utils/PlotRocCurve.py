import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_scores, pos_label=1):
    # 计算 ROC 曲线和 AUC 评分
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

# 示例数据
# y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
# y_scores = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.6, 0.3, 0.5, 0.8, 0.4])

# 绘制 ROC 曲线
# plot_roc_curve(y_true, y_scores)