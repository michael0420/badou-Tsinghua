#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.samples = X.shape[0]
        # 零均值化
        X = X-X.mean(axis=0)
        # 求协方差矩阵
        self.covariance = X.T.dot(X)/(self.samples-1)
        # 求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components = eig_vectors[idx[:self.n_components]].T # 注意要转置才能使特征向量按列排
        # 对X进行降维
        return X.dot(self.components)

if __name__ == '__main__':
    pca = PCA(n_components=2)
    X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
    newX = pca.fit_transform(X)
    print(newX)