import numpy as np

def impl_pca(X, K):
    """
    实现pca降维算法
    :param X: 待降维的矩阵
    :param K: 目标维度
    :return: 降维后的矩阵
    """
    # 1.对原始数据零均值化（中心化）：每个维度的值减去该维度的均值
    mean = np.mean(X, axis=0)  # x方向的均值，即每列的均值
    centrX = X - mean
    print("中心化后的矩阵:\n", centrX)
    # 2.求协方差矩阵
    m = centrX.shape[0]
    cov = np.dot(centrX.T, centrX) / (m - 1)
    # 3.对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
    a, b = np.linalg.eig(cov) # a是特征值，b是特征向量
    print("特征值为：\n", a)
    print("特征向量为：\n", b)
    # 求特征值中topK的索引
    index = np.argsort(-1 * a)
    # topK特征值对应的特征向量分别作为列向量组成特征向量矩阵
    U = b[:, index[:K]]
    print("%d阶降维转换矩阵:\n" % K,U)
    # 4.将数据集投影到选取的特征向量上
    Z = np.dot(X, U)
    print("降维后矩阵为:\n", Z)
    return Z

if __name__ == '__main__':
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
    K = X.shape[1] - 1
    Z = impl_pca(X, K)


