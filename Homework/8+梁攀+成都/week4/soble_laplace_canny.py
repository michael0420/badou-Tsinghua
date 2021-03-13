import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("cat.png")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny_img = cv2.Canny(gray_img, 100, 150)
'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''
soble_img_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = 3) #对x求导
soble_img_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = 3) #对y求导
laplace_img = cv2.Laplacian(gray_img, cv2.CV_64F, ksize = 7)

plt.subplot(231)
plt.imshow(gray_img, "gray")
plt.title("Original")

plt.subplot(232)
plt.imshow(soble_img_x, "gray")
plt.title("Sobel_x")

plt.subplot(233),plt.imshow(soble_img_y, "gray"),plt.title("Sobel_y")
plt.subplot(234),plt.imshow(laplace_img,  "gray"),plt.title("Laplace")
plt.subplot(235),plt.imshow(canny_img, "gray"),plt.title("Canny")
plt.show()
