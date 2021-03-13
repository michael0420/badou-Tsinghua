#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
使用PCA求样本矩阵的K阶降维矩阵
'''
import numpy as np

class PCA:
    '''
    PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self,X,K):
        '''
        :param X: 训练样本矩阵X
        :param K: X的降维矩阵的阶数
        '''
        self.X = X  # 样本矩阵
        self.K = K  # 要降到的维数
        self.centrX = self._centralized() # 样本矩阵做零均值化后的矩阵
        self.C = self._cov() # 样本的协方差矩阵
        self.U = self._U() # 新的标准正交基构成的矩阵
        self.Z = self._Z() # 样本矩阵的降维矩阵

    def _centralized(self):
        '''样本矩阵X的零均值化'''
        print('样本矩阵X:\n', self.X)
        mean = self.X.mean(axis=0) # 竖直方向求均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean # 利用numpy数组的广播特性，实现样本集的零均值化
        print('样本矩阵X的零均值化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵'''
        # 样本集的样例总数
        ns = self.centrX.shape[0]
        # 样本矩阵的协方差矩阵C
        C = self.centrX.T.dot(self.centrX)/(ns-1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, 即新的标准正交基构成的矩阵，shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-a) # 默认从小到大排取索引，加了负号后从大到小排取索引
        # 构建K阶降维的降维转换矩阵U,即新的标准正交基构成的矩阵
        UT = b[ind[:self.K]] # 前k大的特征值对应的特征向量按行排列
        U = np.transpose(UT) # 实现特征向量按列排列
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), m是样本总数，k是降维矩阵中特征维度总数'''
        Z = self.X.dot(self.U) # 这里计算投影的时候用的是样本原始矩阵，应该用样本零均值化后的矩阵更好
        # Z = self.centrX.dot(self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = X.shape[1]-1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)