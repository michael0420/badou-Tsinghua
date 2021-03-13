# encoding=gbk
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
# 获取灰度图像
img = cv2.imread('lenna.png', 1)
# cv2.imshow("src img",img)
# cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray img",gray)
# cv2.waitKey()

# 原灰度图像直方图
plt.figure()
plt.subplot(211)
plt.hist(gray.ravel(), 256) # ravel()将多维数组转换为一维数组
plt.title("原灰度图像直方图")
# 灰度图直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0,256])
plt.subplot(212)
plt.hist(dst.ravel(), 256)
plt.title("直方图均衡化后直方图")
plt.show()

'''
np.vstack():在竖直方向上堆叠
np.hstack():在水平方向上平铺
'''
# cv2.imshow("histogram equalization", np.hstack([gray, dst]))
# cv2.waitKey()

# 彩色图像直方图均衡化
(b, g, r) = cv2.split(img)
b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)
# 合并每一个通道
dst_img = cv2.merge((b_hist,g_hist, r_hist))
cv2.namedWindow("src_rgb")
cv2.namedWindow("dst_rgb")
cv2.imshow("src_rgb", img)
cv2.imshow("dst_rgb", dst_img)
cv2.waitKey()
cv2.destroyAllWindows()