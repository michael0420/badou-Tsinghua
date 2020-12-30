
import cv2
import numpy as np
import math

def function(img):
    height,width,channels =img.shape
    print(img.shape)
    emptyImage=np.zeros((800,800,channels),np.uint8)
    print(emptyImage.shape)
    sh=800/height
    sw=800/width
    print(sh)
    print(sw)
    for i in range(799):
        for j in range(799):
            # x=int(i/sh)
            # y=int(j/sw)   # 向下取整
            # x = round(i/sh)
            # y = round(j/sw)   # 四舍五入
            x = math.ceil(i/sh)
            y = math.ceil(j/sw)   # 向上取整
            emptyImage[i,j]=img[x,y]  # 点的像素值
            # print(img[x, y])
    print(emptyImage[i, j].shape)
    return emptyImage

img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

