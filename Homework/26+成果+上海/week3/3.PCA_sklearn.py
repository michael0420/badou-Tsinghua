# -*- coding: utf-8 -*-

'''
@author: chengguo
Theme：使用第三方库sklearn实现特征提取
'''

import numpy as np
from sklearn.decomposition import PCA

X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
pca=PCA(n_components=2)                    #设定降为2维
# PCA(copy=True, n_components=2, whiten=False)
pca.fit(X)                      #训练
newX=pca.fit_transform(X)       #降维后的数据
print(pca.explained_variance_ratio_)  #贡献率
print(newX)                     #输出




