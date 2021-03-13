#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from decimal import Decimal # 实现一般意义上的四舍五入
import os
print(os.getcwd()) # 获取当前工作路径
def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),dtype=np.uint8) # 创建空数组,np.uint8类型用于存储图像，表示的范围[0-255]
    sh = 800/height # 行方向上缩放比例
    sw = 800/width # 列方向上缩放比例
    for i in range(800):
        for j in range(800):
            # 相对于原始图像虚拟像素点的坐标
            x = int(Decimal(str(i/sh)).quantize(Decimal('1.'),rounding='ROUND_HALF_UP'))
            y = int(Decimal(str(j/sw)).quantize(Decimal('1.'),rounding='ROUND_HALF_UP'))
            emptyImage[i,j] = img[x,y]
    return emptyImage

# 切换当前工作路径为当前文件所在的父目录:
os.chdir(os.path.dirname(os.path.abspath(__file__))) # 这行代码在vscode里必须要加
img = cv2.imread("lenna.png") # 注意:路径中不能出现中文
# img = cv2.imdecode(np.fromfile("lenna.png",dtype = np.uint8),-1)
zoom = function(img)
print(zoom.shape)
cv2.imshow('nearest interp',zoom)
cv2.imshow('image',img)
cv2.waitKey(0) # 这行代码必须要
# cv2.waitKey(0): 是一个和键盘绑定的函数，它的作用是等待一个键盘的输入（因为我们创建的图片窗口
# 如果没有这个函数的话会闪一下就消失了，所以如果需要让它持久输出，我们可以使用该函数）
cv2.destroyAllWindows() #  销毁我们创建的所有窗口，这行代码非必要
