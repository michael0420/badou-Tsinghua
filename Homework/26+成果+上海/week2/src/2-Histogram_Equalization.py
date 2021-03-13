# -*- coding: utf-8 -*-

"""
@author: chengguo
Theme:直方图均衡化-equalizeHist
"""

"""
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可

calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#step1、彩图转化为灰度图
img=cv2.imread("../res/lenna.png",1)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#灰度图的直方图
hist=cv2.calcHist([gray_img],[0],None,[256],[0,256])

#step2、灰度图直方图均衡化
dst=cv2.equalizeHist(gray_img)

plt.figure()                 #新建figure对象
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst]))
cv2.waitKey(0)

"""
np.vstack():在竖直方向上堆叠
np.hstack():在水平方向上平铺
"""




