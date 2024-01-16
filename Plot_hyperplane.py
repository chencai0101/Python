import matplotlib.pyplot as plt
import numpy as np

# 绘制三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_image = np.array([2, 4, 2, 4, 6])
y_image = np.array([3, 5, 7, 5, 9])
z_image = np.array([1, 2, 3, 4, 5])

ax.scatter(x_image, y_image, z_image, c=z_image, cmap='viridis')  # 使用 z 值作为颜色映射

ax.set_xlabel('X')            # X轴的名字
ax.set_ylabel('Y')            # y轴的名字
ax.set_zlabel('Z')            # z轴的名字

plt.show()