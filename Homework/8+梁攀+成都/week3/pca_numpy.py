import numpy as np

class NP_PCA():
    def __init__(self, K):
        self.K = K

    def redu_dim(self, X):
        X = X -X.mean(axis = 0)   #均值化
        cov = np.dot(X.T, X)/X.shape[0]  #协方差
        eig_vals, eig_vectors = np.linalg.eig(cov) #求特征值和向量
        idx = np.argsort(-eig_vals)   #降序排序生成索引
        trans = eig_vectors[:, idx[:self.K]] #前K列
        return np.dot(X, trans)              #降维结果

pca = NP_PCA(2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1],
              [-3,8,45,-2], [1,9,36,1],
              [2,10,62,1], [3,5,83,2]])
new = pca.redu_dim(X)
print(new)