import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread("lenna.png")  #opencv 读取图像
cv2.imshow("Original", image)

chans = cv2.split(image)  #三个通道的像素
colors = ("b", "g", "r")  #三个通道的顺序是BGR

#画一幅图
# plt.figure()     #图的框架
# plt.title("扁平化颜色直方图")      #图的标题
# plt.xlabel("Bins")
# plt.ylabel("pixels")

'''
通道已经分隔
for (chan, color) in zip(chans, colors): #打包为元组
    hist = cv2.calcHist([chan],[0],None,[256],[0,256]) #计算图像的一个通道的直方图
    plt.plot(hist,color = color)
    plt.xlim(0,256)
'''

#通道未分隔
# for chanel,color in enumerate(colors):
#     hist = cv2.calcHist([image],[chanel],None,[256],[0,256])
#     plt.plot(hist, color=color)
#     plt.xlim(0, 256)
# plt.show()

#每个通道单独均衡化
# bh = cv2.equalizeHist(chans[0])
# gh = cv2.equalizeHist(chans[1])
# rh = cv2.equalizeHist(chans[2])

# hist = cv2.calcHist([bh],[0],None,[256],[0,256])
# plt.plot(hist, color='r')
# plt.xlim(0, 256)
# plt.show()

# re = cv2.merge((bh, gh, rh))
# cv2.imshow("result img", re)

#我写的算法
def hist_equalize(img, h, w):
    re_hist = np.zeros(256)
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)    #必须加上类型
    hist = cv2.calcHist([img], [0], None, [255], [0,255])  #先计算一个通道的直方图
    pixes = h * w * 1.
    re_hist[0] = hist[0] / pixes * 255
    re = hist[0]
    for i in range(1, 255):
        re = hist[i] + re
        re_hist[i] = re * 255 / pixes

    for x in range(512):
        for y in range(512):
            out[x,y] = re_hist[img[x,y]]

    return out


h, w ,ch = image.shape
my_bh = hist_equalize(chans[0], h, w)
my_gh = hist_equalize(chans[1], h, w)
my_rh = hist_equalize(chans[2], h, w)

#更简洁的算法
def hist_equal(img, z_max=255):
    H, W = img.shape
    S = H * W * 1.

    out = img.copy()

    sum_h = 0.

    for i in range(1, 255):
        ind = np.where(img == i)   #像素点为i的坐标有哪些 tuple类型
        sum_h += len(img[ind])    #求出这些坐标对应的值，值构成了数组，求数组的长度就知道了等于这个像素点值的像素有多少个
        z_prime = sum_h * z_max / S
        out[ind] = z_prime       #将变换后的值填充回去

    out = out.astype(np.uint8)
    return out

# my_bh = hist_equal(chans[0])
# my_gh = hist_equal(chans[1])
# my_rh = hist_equal(chans[2])
my_re = cv2.merge((my_bh, my_gh, my_rh))
cv2.imshow("result img", my_re)

cv2.waitKey(0)