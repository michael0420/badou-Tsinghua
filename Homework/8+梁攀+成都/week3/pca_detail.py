import numpy as np

class DE_PCA():
    def __init__(self, X, K):   #类的实例化，X样本矩阵，降维之后的输出维度
        self.X = X
        self.K = K
        self.Xcenter = [] #均值化之后的矩阵
        self.Xcov = []    #协方差矩阵
        self.Xtran = []   #特征向量构成的降维转换矩阵
        self.Xresult = [] #降维之后的矩阵

        self.Xcenter = self.centralized()  #均值化
        self.Xcov = self.cov()             #求协方差矩阵
        self.Xtran = self.transfrm()       #求变换矩阵
        self.Xresult = self.redu_dim()     #求降维之后的结果

    def centralized(self):
        centrX = []
        mean = np.mean(self.X, axis = 0)  #按列求均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def cov(self):
        #计算样本数
        ns = np.shape(self.Xcenter)[0]
        #计算协方差
        cov = np.dot(self.Xcenter.T, self.Xcenter) / ns
        print('样本矩阵X的协方差矩阵:\n', cov)
        return  cov

    def transfrm(self):
        a = np.linalg.eig(self.Xcov)
        print('样本集的协方差矩阵的特征值:\n', a[0])
        print('样本集的协方差矩阵的特征向量:\n',a[1])
        ind = np.argsort(-a[0])  #计算排序之后对应的索引
        print("ind:", ind)
        U = [a[1][:, ind[i]] for i in range(self.K)] #取出前K列
        U = np.transpose(U)
        print('%d阶降维转换矩阵:\n' % self.K, U)
        return U

    def redu_dim(self):
        Z = np.dot(self.X, self.Xtran)
        print('样本矩阵X的降维矩阵:\n', Z)
        return Z


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
print('样本集:\n', X)
pca = DE_PCA(X, K)
