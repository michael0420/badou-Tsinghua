# -*- encoding=UTF-8 -*-
import cv2
import numpy as np

#作图片灰度化
grayImg = cv2.cvtColor(cv2.imread("images/lenna.png"),cv2.COLOR_BGR2GRAY)

#作Canny
#def Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None): # real signature unknown; restored from __doc__
cannyImg = cv2.Canny(grayImg,100,800,apertureSize=5)
cv2.imshow("cannyImg",cannyImg)
cv2.waitKey(0)
cv2.destoryAllWindows()

