#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
img = cv2.imread('photo1.jpg')
result3 = img.copy()

# img = cv2.GaussianBlur(img,(3,3),0) # 高斯滤波,用来降噪
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度化，边缘检测要求输入灰度图像
# edges= cv2.Canny(gray,50,150,apertureSize=3) # Canny检测边缘，方便找用于透视变换的顶点
# cv2.imwrite('canny.jpg',edges) # 保存图像
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207,151],[517,285],[17,601],[343,731]]) # 原始图像的4个顶点坐标，可以用画图工具或matplotlib来找
import math
width = int(math.sqrt((207-517)**2+(151-285)**2)) # 337
height = int(math.sqrt((207-17)**2+(151-601)**2)) # 488
print(f'变换后图像的高度:{height},宽度:{width}')
dst = np.float32([[0,0],[337,0],[0,488],[337,488]]) # 目标图像的4个顶点坐标
# 生成透视变换矩阵，进行透视变换
m = cv2.getPerspectiveTransform(src,dst)
result = cv2.warpPerspective(result3,m,(337,488)) # (337,488)为目标图像尺寸(宽度,高度)
cv2.imshow('src',img)
cv2.imshow('result',result)
cv2.waitKey(0)