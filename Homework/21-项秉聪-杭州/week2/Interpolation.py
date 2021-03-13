# -*- encoding=UTF-8 -*-
import cv2
import numpy as np

#读取图片
originalImg = cv2.imread("lenna.png")
src_h,src_w,channels = originalImg.shape
dst_h = int(src_w * 1.5)
dst_w = int(src_h * 1.5)
newImg = np.zeros((dst_h,dst_w,channels),dtype=np.uint8)

#双线性插值
scale_x = float(src_w) / dst_w
scale_y = float(src_h) / dst_h
for chan in range(channels):
    for dst_x in range(dst_h):
        for dst_y in range(dst_w):
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5

            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1,src_w - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1,src_h - 1)

            tempX1 = originalImg[src_y0,src_x0,chan] * (src_x1 - src_x) + originalImg[src_y0,src_x1,chan] * (src_x - src_x0)
            tempX2 = originalImg[src_y1,src_x0,chan] * (src_x1 - src_x) + originalImg[src_y1,src_x1,chan] * (src_x - src_x0)

            newImg[dst_y,dst_x,chan] = tempX1 * (src_y1 - src_y) + tempX2 * (src_y - src_y0)
cv2.imshow('newImg1',newImg)

#邻近性插值
for dst_y in range(dst_h):
    for dst_x in range(dst_w):
        src_x = int(dst_x * scale_x)
        src_y = int(dst_y * scale_y)

        newImg[dst_y,dst_x] = originalImg[src_y,src_x]

cv2.imshow('newImg2',newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

