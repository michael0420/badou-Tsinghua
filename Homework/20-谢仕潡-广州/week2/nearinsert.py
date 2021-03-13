# -*- coding: utf-8 -*-

"""实现最邻近插值"""

import cv2
import numpy as np

img = cv2.imread("lenna.png")

height,width,channels =img.shape

empty_image = np.zeros((1000, 1000, channels), np.uint8)

for i in range(1000):
    for j in range(1000):
        x = round(width/1000*i)
        y = round(height/1000*j)
        empty_image[i,j] = img[x,y]


cv2.imshow("lod image", img)
cv2.imshow("new image", empty_image)
cv2.waitKey(0)