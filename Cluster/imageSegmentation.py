from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def k_means(X,k):
    model = KMeans(n_clusters=k, random_state=9)
    model.fit(X) # 对 X 的形状要求是：n_simples ,n_features，意思是数据集得是一个二维数组
    labels = model.labels_
    centers = model.cluster_centers_  
    socre = silhouette_score(X, labels)  # 这里计算的是整个数据集的轮廓系数，是所有样本轮廓系数的均值
    return labels,centers,socre

image = imread("Resource/data/xiaoxin.jpg") # 读取图片，返回一个包含每个每个像素的RGB值的三位数组
X = image.reshape(-1,3) # 返回一个聚类算法可以用的二维数组 40000 x 3         

labels,centers,socre = k_means(X,2)               # 每个样本点的标签，八个类别的中心点，轮廓系数
segment_img = centers[labels].reshape(image.shape)# 找到每个标签所对应的中心点，然后reshape还原到图像的结构，图像中就只有八种不同颜色的像素点了 
segment_img = segment_img.astype(int)             # 由于rgb值应当是整数，这里做个转换

plt.imshow(image)
plt.axis('off')        # 关掉坐标轴
plt.show()