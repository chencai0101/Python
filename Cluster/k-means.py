import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def k_means(X,k):
    model = KMeans(n_clusters=k, random_state=9)
    model.fit(X) # 对 X 的形状要求是：n_simples ,n_features，意思是数据集得是一个二维数组
    labels = model.labels_
    centers = model.cluster_centers_  
    socre = silhouette_score(X, labels)  # 这里计算的是整个数据集的轮廓系数，是所有样本轮廓系数的均值
    return labels,centers,socre

# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], random_state =9)

# 进行训练
labels,centers,socre = k_means(X,4)

print("轮廓系数:",socre)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], color='red')

plt.show()