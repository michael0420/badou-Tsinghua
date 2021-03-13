
import numpy as np

if __name__ == '__main__':
    #原数据
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
    K = 2
    # 中心化
    mean = np.array([np.mean(y) for y in X.T])
    print('mean:\n', mean)
    cen = []
    cen = X - mean
    print('中心化:\n', cen)
    # 协方差矩阵
    n = np.shape(cen)[0]
    C = np.dot(cen.T, cen) / (n - 1)
    print('协方差矩阵:\n', C)
    # 特征值、特征向量
    a, b = np.linalg.eig(C)
    level = np.argsort(-1 * a)
    print('特征值:\n', a)
    print('特征向量:\n', b)
    print('排序:\n', level)
    # 特征向量和特征值求的转换矩阵
    UT = [b[:, level[i]] for i in range(K)]
    U = np.transpose(UT)
    # 根据转换矩阵求降维后的矩阵Z = XU
    Z = np.dot(X, U)
    print('矩阵:\n', Z)
