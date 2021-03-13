# -*- coding: utf-8 -*-

'''
@author: chengguo
Theme：PCA算法进行特征提取的具体实现：
1、对原始数据进行零均值化          变量减去均值
2、求协方差矩阵                   D=Z(T)*Z/m
3、对协方差矩阵求特征值和特征向量，然后将特征向量组成新的特征空间     使用 np.linalg.eig
'''

import numpy as np

class PCA_detail(object):

    '求样本矩阵X的K阶降维矩阵Z'
    def __init__(self,X,K):
        self.X=X         #样本矩阵X
        self.K=K         #K阶降维矩阵的K值
        self.centX=[]    #X的中心化矩阵
        self.C=[]        #X的协方差矩阵
        self.U=[]        #X的降维转换矩阵
        self.Z=[]        #X的降维矩阵Z

        self.centX=self._cent()
        self.C=self._cov()
        self.U=self._U()
        self.Z=self._Z()


    '对原始数据进行零均值化(中心化)。每个变量减去它的均值，即平移，使平移后所有数据的中心点为(0,0)'
    def _cent(self):
        print('3、输入的样本矩阵X：\n',self.X)
        centX=[]
        mean=np.array([np.mean(attr) for attr in self.X.T])       #样本集的特征均值，T代表转置矩阵
        centX=self.X-mean
        print('4、样本X的特征均值：\n',mean)
        print('5、样本X的中心化:\n',centX)
        return centX


    '求中心化后的协方差矩阵C   （公式在PPT第29页）'
    def _cov(self):
        '样例总数'
        ns=np.shape(self.centX)[0]
        '求协方差矩阵（如果做了中心化，那么协方差矩阵 D=Z(T)*Z/(1/M)，需要注意顺序转置矩阵乘以原矩阵'
        C=np.dot(self.centX.T,self.centX)/(ns-1)
        print('6、样本的协方差矩阵:\n',C)
        return C


    '求X的降维转换矩阵U'
    def _U(self):
        '求出特征值和特征向量（PPT第30页）'
        a,b=np.linalg.eig(self.C)
        '''将--特征值--由大到小进行排序，选出其他其中最大的K个，将其对应的K个特征向量分别作为列向量组成特征向量矩阵w(nxk)'''
        index=np.argsort(-1*a)        #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
        '构建K阶降维矩阵的转置矩阵U'
        UT=[b[:,index[i]] for i in range(self.K)]
        U=np.transpose(UT)            #transpose在不指定参数是默认是矩阵转置
        print('9、%d阶降维转换矩阵U:\n'%self.K, U)
        return U


    '待输出的降维矩阵Z'
    def _Z(self):
        '根据Z=XU计算降维矩阵'
        Z=np.dot(self.X,self.U)
        print('10、X Shape：',np.shape(self.X))
        print('11、U Shape：', np.shape(self.U))
        print('12、Z Shape：', np.shape(Z))
        print('13、Z', Z)
        return Z


if __name__ == '__main__':
    #输入一个样本集x，行是样例，列是特征维度
    '10个样本，维度为3'
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
    K=np.shape(X)[1]-1
    print('1、样本集(10行3列，10个样例，每个样例3个特征)',X)
    print('2、样本的特征维度：\n',K)
    pca=PCA_detail(X,K)

