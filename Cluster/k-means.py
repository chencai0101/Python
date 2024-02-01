import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def k_means(X,k):
    model = KMeans(n_clusters=k, random_state=9)
    y_pred = model.fit_predict(X)  # 不需要 y 参数，因为它会在训练过程中自动预测目标变量。
    centers = model.cluster_centers_  
    socre = silhouette_score(X, model.labels_)  # 这里计算的是整个数据集的轮廓系数，是所有样本轮廓系数的均值
    return y_pred,centers,socre

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state =9)

# 进行训练
y_pred,centers,socre = k_means(X,4)

print("轮廓系数:",socre)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(centers[:, 0], centers[:, 1], color='red')

plt.show()