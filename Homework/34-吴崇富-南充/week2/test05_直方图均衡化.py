#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
# 切换当前工作路径为当前文件所在的父目录:
os.chdir(os.path.dirname(os.path.abspath(__file__))) # 这行代码在vscode里必须要加
# 获取图像并灰度化
img = cv2.imread('lenna.png',1) # 以三通道方式读图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度化
# cv2.imshow('img_gray',gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
# cv2.imshow('Histogram Equalization',dst)
cv2.imshow('Histogram Equalization',np.hstack([gray,dst])) # 第二个参数对两个图像水平拼接是为了看出对比效果
# cv2.imshow('Histogram Equalization',np.vstack([gray,dst]))

# 显示直方图
plt.rcParams['font.sans-serif'] = [u'SimHei']  # 展示中文
plt.rcParams['axes.unicode_minus'] = False  # 使用非unicode的负号，当使用中文时要设置
# plt.figure() # 添加画布,这里非必须
hist1 = cv2.calcHist([gray],[0],None,[256],[0,255])
hist2 = cv2.calcHist([dst],[0],None,[256],[0,255])
plt.subplot(211)
plt.plot(hist1)
plt.title('直方图均衡化前')
plt.xlabel('灰度值')
plt.ylabel('像素点个数')
plt.xlim([0,255])

plt.subplot(212)
plt.plot(hist2)
plt.title('直方图均衡化后')
plt.xlabel('灰度值')
plt.ylabel('像素点个数')
plt.xlim([0,255])
plt.tight_layout(2,rect=(0,0,1,0.97)) # 调整子图 参数:画布边缘以及子图之间的间距,rect:tuple (left, bottom, right, top),
plt.suptitle(u'直方图均衡化的效果',fontsize=18) # 添加主标题
plt.show()


# cv2.imshow('src',img)
# 彩色图像的直方图均衡化，需要分解通道，对每一个通道均衡化
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
dst = cv2.merge((bH,gH,rH))
# cv2.imshow('dst_rgb',dst)
cv2.imshow('rgb_histogram_equalization',np.hstack((img,dst)))

cv2.waitKey(0)
