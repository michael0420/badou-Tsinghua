# -*- coding: utf-8 -*-

'''
@author: chengguo
Theme：使用canny算子进行边缘检测
'''

'''
edge = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
image：源图像，该图像必须为单通道的灰度图
threshold1：阈值1 (minVal)
threshold2：阈值2 (maxVal)
apertureSize：可选参数，Sobel算子的大小 (卷积核大小)
L2gradient 参数设定求梯度大小的方程
'''

import cv2

img=cv2.imread('lenna.png',1)
gary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("demo",cv2.Canny(gary,200,300))
cv2.waitKey(0)
cv2.destroyAllWindows()





