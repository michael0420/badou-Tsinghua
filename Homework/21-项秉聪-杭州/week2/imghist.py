# -*- encoding=UTF-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#change to img dir
#os.chdir("E:\\python project\\badou_study\\class002\\homework\\")
colorImg = cv2.imread("lenna.png")
b,g,r = cv2.split(colorImg)
bHist = cv2.equalizeHist(b)
gHist = cv2.equalizeHist(g)
rHist = cv2.equalizeHist(r)
colorHistImg = cv2.merge((bHist,gHist,rHist))
#彩色图均衡化
imgHstack = np.hstack((colorImg,colorHistImg))
cv2.imshow("color show",imgHstack)

#灰度图均衡化
grayImg = cv2.cvtColor(colorImg,cv2.COLOR_BGR2GRAY)
grayHistImg = cv2.equalizeHist(grayImg)
imgHstack2 = np.hstack((grayImg,grayHistImg))
cv2.imshow("gray show",imgHstack2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#彩色图直方图
colors = ('b','g','r')
plt.figure("color Hist")
plt.title("color Hist")
plt.xlabel("Bins")
plt.ylabel("pixels")
plt.xlim([0,256])
for (i,j) in zip(colors,(b,g,r)):
    tmphist = cv2.calcHist([j],[0],None,[256],[0,256])
    plt.plot(tmphist,color=i)
plt.show()

plt.figure("color equalization Hist")
plt.title("color equalization Hist")
plt.xlabel("Bins")
plt.ylabel("pixels")
plt.xlim([0,256])
for (i,j) in zip(colors,(bHist,gHist,rHist)):
		tmphist = cv2.calcHist([j],[0],None,[256],[0,256])
		plt.plot(tmphist,color=i)
plt.show()

#灰度图直方图
grayhist1 = cv2.calcHist([grayImg],[0],None,[256],[0,256])
grayhist2 = cv2.calcHist([grayHistImg],[0],None,[256],[0,256])
plt.figure("gray Hist")
plt.title("gray Hist")
plt.xlabel("Bins")
plt.ylabel("pixels")
plt.plot(grayhist1)
plt.plot(grayhist2)
plt.xlim([0,256])
plt.show()
