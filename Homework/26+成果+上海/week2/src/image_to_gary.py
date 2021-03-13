# -*- coding: utf-8 -*-

"""
@author: chengguo
彩色图像的灰度化、二值化
"""

import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
import cv2

plt.subplot(221)                               #创建一个2*2的图像，1是在第一个区域
img=plt.imread("../res/lenna.png")             #加载图片资源
#img=cv2.imread("../res/lenna.png",False)
plt.imshow(img)                                #plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示
print("---image lenna----")
print(img)

#step1、灰度化
plt.subplot(222)
img_gray=rgb2gray(img)                         #彩色图转化为灰度图
#img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray,cmap="gray")
print("---image gray----")
print(img_gray)


#step2、二值化（非黑即白）
"""
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j]<=0.5:
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
"""

img_binary=np.where(img_gray>0.5,1,0)    #转化为二值图，非黑即白
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

#print(img.shape)    #参数-1为按原通道读入，不写的话默认读入三通道图片，例如（112，112，3）
#print(img.shape[0]) #读入的时图片的高度height
#print(img.shape[1]) #读入的时图片的宽度weight

plt.subplot(223)
plt.imshow(img_binary,cmap="gray")
plt.show()










