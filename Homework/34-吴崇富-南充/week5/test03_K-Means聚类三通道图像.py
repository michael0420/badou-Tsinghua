#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

'''
在OpenCV中，Kmeans()函数原型如下所示：
compactness,labels,centers =  = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，预设的分类标签或者None,用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，三种方法是：
        cv2.KMEANS_PP_CENTERS：使用k-means++选择初始中心点);
        cv2.KMEANS_RANDOM_CENTERS：随机选择中心点；
        cv2.KMEANS_USE_INITIAL_LABELS: 第一次迭代使用提供的标签来计算中心。之后的迭代使用随机或半随机中心(使用上面的两个标志)。
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
返回值：
	compactness：紧密度，返回每个点到相应重心的距离的平方和
	labels：结果标记，每个成员被标记为0,1等
	centers：由聚类中心像素值组成的数组
'''
#读取原始图像
img = cv2.imread('lenna.png') # 第二个参数默认是1，表示以三通道方式读图
# print(img.shape) # (512, 512, 3)

#图像二维像素转换为一维,转换后一个通道的像素值为1列，方便聚类
data = img.reshape((-1,3))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1)

# 初始中心点选择设置
flags = cv2.KMEANS_RANDOM_CENTERS # 随机选择初始中心点
# flags = cv2.KMEANS_PP_CENTERS # 使用k-means++选择初始中心点

# K-Means聚类，聚集成2类、4类、8类、16类、64类
cluster_results = []
for i in range(1,7):
    if i == 5:
        continue
    cluster_results.append(cv2.kmeans(data,2**i,None,criteria,10,flags))

images = []
for compactness,labels,centers in cluster_results:
    # 图像转换回uint8二维类型
    centers_ = np.uint8(centers) # 得到质心的灰度值
    # print(centers,centers.shape)
    # print(centers_,centers_.shape)
    # 利用类别标签数组作为质心数组的索引(最好扁平化),实现簇内每个像素点(标签数组的长度等于像素点个数)的灰度值被质心的灰度值(质心数组[索引号])替换
    res = centers_[labels.flatten()] # 这句代码很关键，返回(像素点个数,通道数)的二维数组
    # print(labels.shape) #(262144, 1)
    # print(res,res.shape) #(262144, 3)
    dst = res.reshape(img.shape) # 生成最终图像
    # 图像转换为RGB显示
    dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB) # 这行代码必须加，否则在plt中显示的颜色不正常
    images.append(dst)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
#显示图像
titles = ['原始图像','聚类图像 k=2','聚类图像 k=4','聚类图像 k=8','聚类图像 k=16','聚类图像 k=64']
images.insert(0,cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # 注意原图像要转换为RGB方式才能用plt正确显示
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    # plt.xticks([]),plt.yticks([]) # 关闭刻度显示
    # 上面代码等价于
    plt.axis('off')
plt.show()

# 验证整数数组作为索引,理解res = centers_[labels.flatten()]这行代码
# A = np.arange(10).reshape(2,5)
# B = np.array([0,1]*5).reshape(-1,1)
# print(A,A[B.flatten()])
# print(A.shape,A[B.flatten()].shape)