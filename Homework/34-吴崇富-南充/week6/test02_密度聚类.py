#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import pandas as pd

iris = load_iris() # 导入鸢尾花数据集
# X = pd.DataFrame(iris.data)
# X.columns = iris.feature_names # iris的特征名称作为列名
X = iris.data
print(X)

# 密度聚类
dbscan = DBSCAN(eps=0.4,min_samples=9) # 两个参数分别对应ε和minPts
dbscan.fit(X)
label_pred = dbscan.labels_ # 给出类别号
print(label_pred)

# 绘散点图
# colors = 'rgbk'
colors = 'rgb'
# markers = 'o*+'
markers ='o*+'
for i in range(3):
    plt.scatter(X[label_pred == i][:,0],X[label_pred == i][:,1],c=colors[i],marker=markers[i],label='label'+str(i))
plt.xlabel('septal length') # x轴标签：花萼长度,来自于DataFrame
plt.ylabel('septal width') # y轴标签:花萼宽度,来自于DataFrame
plt.legend(loc=2) # 设置左上角图例
plt.show()

