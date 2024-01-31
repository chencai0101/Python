import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def k_means(X,k):
    model = KMeans(n_clusters=k, random_state=9)
    y_pred = model.fit_predict(X)  # 不需要 y 参数，因为它会在训练过程中自动预测目标变量。
    centers = model.cluster_centers_
    return y_pred,centers

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state =9)

# 进行训练
y_pred,centers = k_means(X,4)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(centers[:, 0], centers[:, 1], color='red')

plt.show()