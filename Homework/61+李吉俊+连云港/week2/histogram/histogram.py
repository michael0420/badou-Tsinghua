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


# 灰度图像直方图
'''
# 获取灰度图像
img = cv2.imread("F:/Small instance of algorithm/esb.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)
cv2.waitKey(0)
'''
'''
# 灰度图像的直方图, 方法一

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()                        # 新建一个图像
plt.title("Grayscale Histogram")    # 绘图的标题
plt.xlabel("Bins")                  # X轴标签
plt.ylabel("# of Pixels")           # Y轴标签
plt.plot(hist)                      # 绘制直方图
plt.xlim([0, 256])                  # 设置x坐标轴范围
plt.show()
'''
''' 
# 灰度图像的直方图，方法二
plt.figure()
plt.hist(gray.ravel(), 256)  # hist绘制直方图，gray.ravel()多维数组转成一维数组，256代表灰度级[0, 255]
plt.show()

'''
#彩色图像直方图
image = cv2.imread("F:/badou-assignment/the-first-week/esb.jpg")
cv2.imshow("Original",image)
cv2.waitKey(0)

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):  # 在for循环里zip()函数用来并行遍历列表
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()



