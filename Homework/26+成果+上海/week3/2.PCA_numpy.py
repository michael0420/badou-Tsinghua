# -*- coding: utf-8 -*-

'''
@author: chengguo
Theme：使用numpy实现特征提取
'''

import numpy as np

class PCA():
    def __init__(self,k):
        self.k=k

    def fit_transform(self,X):
        self.features=X.shape[1]    #求输入样本的维度，shape[0]是行数，shape[1]是列数
        #求协方差矩阵
        X=X-X.mean(axis=0)  #均值
        self.cov=np.dot(X.T,X)/X.shape[0]
        #求特征值和特征向量
        vauls,vectors=np.linalg.eig(self.cov)
        #对特征值进行排序
        index=np.argsort(-vauls)
        #求降维矩阵U
        self.U=vectors[:,index[:self.k]]
        #对样本X进行降维
        return np.dot(X,self.U)

    ''' axis
        不设置值，对m * n个数求均值，返回一个实数
        axis = 0：压缩行，对各列求均值，返回1 * n矩阵
        axis = 1 ：压缩列，对各行求均值，返回m * 1矩阵  '''


if __name__ == '__main__':
    pca=PCA(k=2)   #设定降2维
    X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
    newX=pca.fit_transform(X)
    print(newX)