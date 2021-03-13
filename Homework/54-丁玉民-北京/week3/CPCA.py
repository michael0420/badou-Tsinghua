import numpy as np

class CPCA():
    '''
    X：需要降维的样本数据
    k：需要降到的纬度数
    '''

    def __init__(self, X, k):
        self.X = X
        self.k = k

        # 中心化操作后的样本数据
        self.X_centered = self._centralization()
        # 协方差矩阵
        self.Cov_matrix = self._cov_matrix()
        # k个特征向量组成的矩阵
        self.U = self._feature_martix()
        # 降维后的矩阵
        self.Z = self._Z()

    def _centralization(self):
        '''
        对样本数据进行均值中心化处理
        :return: 均值中心化后的数据
        '''
        # 计算每一纬度的均值
        X_mean = np.array([np.mean(self.X[:, c]) for c in range(self.X.shape[1])])
        print('均值向量：', X_mean)
        # 对原始样本数据进行中心化处理，减去均值
        X_centered = self.X - X_mean
        print('均值中心化后的数据：\n', X_centered)
        return X_centered

    def _cov_matrix(self):
        '''
        对中心化处理后的数据求协方差矩阵， (X_centered.T.dot(X_centered))/m
        :return: 协方差矩阵
        '''
        cov_matrix = np.dot(self.X_centered.T, self.X_centered) / (len(self.X_centered) - 1)
        print('协方差矩阵：\n', cov_matrix)
        return cov_matrix

    def _feature_martix(self):
        '''
        对协方差矩阵求特征值和对应的特征向量
        :return: topK特征向量
        '''
        w, U = np.linalg.eig(self.Cov_matrix)
        print('协方差矩阵的特征值：', w)
        print('协方差矩阵的特征向量矩阵：\n', U)
        w_i = np.argsort(-1 * w)
        _U = [U[:, w_i[i]] for i in range(self.k)]
        print(np.transpose(_U))
        print('top ', k, ' 个特征向量矩阵：\n', np.array(_U).T)
        return np.array(_U).T

    def _Z(self):
        '''
        均值中心化处理后的数据集与topK特征向量矩阵点乘，得到降维后的数据集
        :return: 降维后的数据集
        '''
        Z = self.X_centered.dot(self.U)
        print('降维后的数据集：\n', Z)
        return Z

    def pca_use_sklearn(self):
        pass


if __name__ == '__main__':
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

    k = X.shape[1] - 1
    pca = CPCA(X, k)

