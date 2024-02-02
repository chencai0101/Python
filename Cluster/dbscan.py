from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def dbscan(X,eps,min_samples):
    # 定义 DBSCAN 算法的参数
    model = DBSCAN(eps=eps, min_samples=min_samples)
    # 使用 DBSCAN 算法对数据进行聚类
    model.fit(X)
    labels=model.labels_
    socre = silhouette_score(X, labels)  # 这里计算的是整个数据集的轮廓系数，是所有样本轮廓系数的均值
    return labels,socre

# 生成一个包含两个聚类的模拟数据集
X, y = make_moons(n_samples=200, noise=0.05)
labels,socre = dbscan(X,0.3,5)
print("轮廓系数为",socre)

# 打印每个簇的标签，并绘制散点图
plt.scatter(X[:,0],X[:,1],c=labels)
plt.show()