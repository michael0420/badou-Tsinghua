#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；低阈值
第三个参数是滞后阈值2。高阈值
'''

img = cv2.imread('lenna.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度化
cv2.imshow('canny',cv2.Canny(gray,200,300))
cv2.waitKey()
cv2.destroyAllWindows()
