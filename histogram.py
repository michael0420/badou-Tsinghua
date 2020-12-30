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

'''
# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()
'''

'''
# 灰度图像的直方图, 方法二
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()
'''


#彩色图像直方图

image = cv2.imread("lenna.png")
print(image)
cv2.imshow("Original",image)
cv2.waitKey(1000)

# Grab the image channels, initialize the tuple of colors and the figure
chans = cv2.split(image)
# print(chans)

# blue, green, red = cv2.split(image)
#
# cv2.imshow('blue', blue)
# cv2.imshow('green', green)
# cv2.imshow('red', red)

# zeros = np.zeros(blue.shape, np.uint8)
# print(zeros)
# print(blue.shape)
"""
blueBGR = cv2.merge((blue, zeros, zeros))
greenBGR = cv2.merge((zeros, green, zeros))
redBGR = cv2.merge((zeros, zeros, red))
cv2.imshow('blue BGR', blueBGR)
cv2.imshow('green BGR', greenBGR)
cv2.imshow('red BGR', redBGR)
"""


# cv2.waitKey(0)
# cv2.destroyAllWindows()

colors = ("b","g","r")
zipped = zip(chans,colors)
print(type(zipped))
print(list(zipped))


plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")


for (chan,color) in zip(chans,colors):
    # Create a histogram for the current channel and plot it
    hist = cv2.calcHist([chan], [0], None,[256],[0,256])
    plt.plot(hist,color = color)
    # plt.hist(chan.ravel(), 256, [0, 256])
    plt.xlim([0,256])
plt.show()

hist = []
for i in range(3):
    result_hist = cv2.calcHist([image],[i],None,[256],[0,256])
    hist.append(result_hist)
    plt.subplot(1, 3, i+1)
    plt.plot(hist[i], color = ["b","g","r"][i])
    # plt.plot(hist[i], color=color)
plt.show()


plt.plot([1, 2, 3])
plt.subplot(211)
plt.show()
