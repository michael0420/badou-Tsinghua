#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。有single、complete、average、weighted、median、centroid、ward
细究距离计算方法，参考https://blog.csdn.net/Haiyang_Duan/article/details/78906058
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
功能: 由给定的连接矩阵定义的层次聚类形成平面聚类 
Form flat clusters from the hierarchical clustering defined by the given linkage matrix
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
3.criterion:实现阈值的方法
'''
'''
scipy.cluster.hierarchy.dendrogram(Z, p=30, truncate_mode=None, color_threshold=None, 
get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False,
show_leaf_counts=True, no_plot=False, no_labels=False, leaf_font_size=None, 
leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, 
ax=None, above_threshold_color='b')
功能:将层次聚类绘制为树状图
第一个参数Z是linkage得到的矩阵
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X,'ward') # 层次聚类,'ward'通过ward方差最小化算法算的距离
# f = fcluster(Z,4,'distance') # 这行代码好像不用也行
fig = plt.figure(figsize=(5,3)) # 设置画布
dn = dendrogram(Z) # 将层次聚类绘制为树状图 怎么提取结果???
print(Z)
# print(f)
print(dn)
plt.show()