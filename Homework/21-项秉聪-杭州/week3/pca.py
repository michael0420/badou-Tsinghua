# -*- encoding=UTF-8 -*-
import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,in_array):
        # mean() 函数定义：
        # numpy.mean(a, axis, dtype, out，keepdims )
        # mean()函数功能：求取均值
        # 经常操作的参数为axis，以m * n矩阵举例：
        # axis 不设置值，对 m * n个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回1 * n矩阵
        # axis = 1 ：压缩列，对各行求均值，返回m * 1矩阵
        zeroArray = in_array.mean(axis=0)
        #进行数据中心化
        in_array = in_array - zeroArray
        print("in_array",in_array)
        #求协方差矩阵
        #np.dot(a,b), 作a、b乘积，也可写作a.dot(b)
        self.covariance = np.dot(in_array.T,in_array) / (in_array.shape[0] - 1)
        print("self.covariance",self.covariance)
        # 函数：numpy.linalg.eig(a)
        # 参数: a：想要计算奇异值和右奇异值的方阵。
        # 返回值：w：特征值。每个特征值根据它的多重性重复。这个数组将是复杂类型，除非虚数部分为0。当传进的参数a是实数时，得到的特征值是实数。
        #       v：特征向量。
        eig_w,eig_v = np.linalg.eig(self.covariance)
        print("eig_w",eig_w)
        print("eig_v",eig_v)
        # 获取降序排列特征值
        idx = np.argsort(-eig_w)
        #生成降维矩阵
        self.components_ = eig_v[:,idx[:self.n_components]]
        #投影一个新的矩阵
        print("in_array",in_array)
        print("self.components_", self.components_)
        return  np.dot(in_array,self.components_)

#inputArray = np.array([[10,6],[11,4],[8,5],[3,3],[2,2.8],[1,1]])
inputArray = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)
outputArray = pca.fit_transform(inputArray)
print("outputArray",outputArray)