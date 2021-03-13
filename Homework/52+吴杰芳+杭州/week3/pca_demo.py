import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets.base import load_iris
import numpy as np
"""pca运用的例子"""
# 加载鸢尾花数据，x表示样本的属性数据，y表示样本标签
x, y = load_iris(return_X_y=True)
print("original_x size:", x.shape)
print("y的取值：", set(y), "y size: ", y.shape)
pca = PCA(n_components=2) # 将到2维
reduced_x = pca.fit_transform(x) # 降维操作
# print("reduced_x:\n", reduced_x)
print("reduced_x size: ", reduced_x.shape)

red_x = reduced_x[np.where(y == 0), 0].T
red_y = reduced_x[np.where(y == 0), 1].T
blue_x = reduced_x[np.where(y == 1), 0].T
blue_y = reduced_x[np.where(y == 1), 1].T
green_x = reduced_x[np.where(y == 2), 0].T
green_y = reduced_x[np.where(y == 2), 1].T
# 降维数据画散点图查看效果
plt.scatter(red_x, red_y,c='r',marker="*")
plt.scatter(blue_x, blue_y,c='b',marker="D")
plt.scatter(green_x, green_y,c='g',marker=".")
plt.savefig("./pca_demo.png")
plt.show()