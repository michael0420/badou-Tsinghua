#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png',0) # 读取灰度图
print(img.shape)

# 获取图像高度、宽度
rows,cols = img.shape

#图像二维像素转换为一维,转换后一个通道的像素值为1列，方便聚类
data = img.reshape((rows*cols,1))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

# 初始中心点选择设置
flags = cv2.KMEANS_RANDOM_CENTERS # 随机选择初始中心点
# flags = cv2.KMEANS_PP_CENTERS # 使用k-means++选择初始中心点

# K-Means聚类 聚集成4类
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)
print(compactness)
print(labels)
print(centers)

# 生成最终图像
dst = labels.reshape((img.shape[0],img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([]) # 关闭刻度显示
plt.show()