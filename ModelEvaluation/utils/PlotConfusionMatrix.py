import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(y_true, y_pred, classes=['0', '1'], title='Confusion matrix', cmap=plt.cm.Blues):
 # 创建混淆矩阵
 cm = confusion_matrix(y_true, y_pred)

 # 绘制混淆矩阵
 plt.imshow(cm, interpolation='nearest', cmap=cmap)
 plt.title(title)
 plt.colorbar()
 tick_marks = np.arange(len(classes))
 plt.xticks(tick_marks, classes, rotation=45)
 plt.yticks(tick_marks, classes)

 fmt = 'd'
 thresh = cm.max() / 2.
 for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
 horizontalalignment="center",
 color="white" if cm[i, j] > thresh else "black")

 plt.tight_layout()
 plt.ylabel('True label')
 plt.xlabel('Predicted label')
 plt.show()

# 示例用法
# 假设有一个二分类问题，类别为 0 和 1
# y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# 调用函数绘制混淆矩阵
# plot_confusion_matrix(y_true, y_pred)
