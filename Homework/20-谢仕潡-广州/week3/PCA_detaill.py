#coding=utf-8

import numpy as np


class PCA(object):

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.Z = self._Z()

    def _centralized(self):
        "矩阵中心化"
        # 1.求每一维(特征)的均值
        mean = [np.mean(i) for i in self.X.T]
        # 2.求出中心化矩阵
        centrX = self.X - mean

        return centrX


    def _cov(self):
        "求协方差"
        # 1.求样本总数
        ns = np.shape(self._centralized())[0]
        # 2.求协方差矩阵
        C = np.dot(self._centralized().T, self._centralized())/(ns-1)
        #返回协方差矩阵
        return C


    def _U(self):
        "求特征值与特征向量,以及由特征向量组成的矩阵"
        C = self._cov()
        # 1.求协方差的特征值与特征向量
        a, b = np.linalg.eig(C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 2.对特征值按索引降序排序
        ind = np.argsort(-1*a)
        # 3.求前K维特征向量组成的矩阵
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)

        return U


    def _Z(self):
        """求解降维后的矩阵"""
        U = self._U()
        Z = np.dot(self.X,U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1

    pca = PCA(X,K)
    print(pca)

