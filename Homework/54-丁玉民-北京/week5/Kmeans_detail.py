import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

def kmeans(data, nums, max_iter, n_init=10):
    '''
    :param data: 原始数据
    :param nums: 聚类数
    :param max_iter: 最大迭代次数
    :return:
    '''

    if len(data) < nums:
        return data

    best_inertia, best_centers = None, None
    for i in range(n_init):
        centers, inertia = kmeans_single(data, nums, max_iter)
        print('当前分类结果：', inertia, '最优分类结果：', best_inertia)
        if best_inertia is None or inertia <= best_inertia:
            best_centers = centers.copy()
            best_inertia = inertia

    return best_centers

def kmeans_single(data, nums, max_iter):
    center_ = {}
    random_center = random.sample(list(data), k=nums)
    labels = [0] * len(data)
    # 设置前k个样本为质心
    for k in range(nums):
        center_[k] = random_center[k]
    print('初始质心：', center_)
    for i in range(max_iter):
        print('max_iter:', i)
        # 保存聚类后的数据，key为聚类的索引
        center_iter = {}
        for k in range(nums):
            center_iter[k] = []

        for index, x in enumerate(data):
            # x到k个质心的距离列表
            distance_arr = []
            for c in center_:
                # 分别计算x到k个质心的距离，并放入列表
                distance_arr.append(getDistance(x, center_[c]))
            # 取距离最小的索引，即聚类的索引
            k_i = distance_arr.index(min(distance_arr))
            center_iter[k_i].append(x)
            labels[index] = k_i

        pre_c = dict(center_)
        # 重新计算当前分类下的质心
        for c in center_:
            center_[c] = np.mean(center_iter[c], axis=0)

        flag = True
        if c in pre_c:
            pre_ = pre_c[c]
            cur_ = center_[c]
            print('上一次质心：', pre_)
            print('当前质心：', cur_)
            if np.sum(pre_) != np.sum(cur_):
                flag = False

        if flag:
            break

    center_list = np.array(list(center_.values()))
    inertia = np.sum((data - center_list[labels]) ** 2, dtype=np.float64)

    return center_, inertia

def getDistance(x1, x2):
    '''
    求样本到质心之间的欧式距离
    :param x1: 样本x1
    :param x2: 样本x2
    :return:
    '''
    return np.sum(np.square(np.array(x1) - np.array(x2)))

def fit(X, center):
    y_pre = [0] * len(X)
    for i, x in enumerate(X):
        distance_arr = []

        for c in center:
            distance_arr.append(getDistance(x, center[c]))
        y_pre[i] = distance_arr.index(min(distance_arr))
    return y_pre


if __name__ == '__main__':
    X = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
         ]
    # X = [[0, 2], [0, 0], [1, 0], [5, 0], [5, 2]]
    center = kmeans(X, 3, 100)
    print(center)
    predit = fit(X, center)
    # n_init:用不同的质心初始化值运行算法的次数
    clf = KMeans(n_clusters=3, init='random')
    ac = AgglomerativeClustering(n_clusters=3)
    dbscan = DBSCAN(eps=0.05, min_samples=6)
    y_hat = clf.fit_predict(X)
    y_hat_ac = ac.fit_predict(X)
    y_hat_dbscan = dbscan.fit_predict(X)
    print(y_hat)
    print(predit)

    x = [i[0] for i in X]
    y = [i[1] for i in X]

    plt.subplot(2, 2, 1)
    plt.scatter(x, y, c=predit, marker='D')
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(['A', 'B', 'C'])
    plt.title('detail')
    plt.subplot(2, 2, 2)
    plt.scatter(x, y, c=y_hat, marker='D')
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(['A', 'B', 'C'])
    plt.title('sk kmeans')
    plt.subplot(2, 2, 3)
    plt.scatter(x, y, c=y_hat_ac, marker='D')
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(['A', 'B', 'C'])
    plt.title('Agglomerative')
    plt.subplot(2, 2, 4)
    plt.scatter(x, y, c=y_hat_dbscan, marker='D')
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(['A', 'B', 'C'])
    plt.title('dbscan')
    plt.show()
