# -*- coding: utf-8 -*-

"""实现双线性插值"""

import cv2
import numpy as np


def bilinnear_inter(img,new_height,new_width):
    # 1.读取图像shape
    heigh,width,channels = img.shape
    # 2.构造零像素点矩阵
    empty_image = np.zeros((new_height,new_width, channels), dtype=np.uint8)
    # 3.目标图像映射回原图像
    for j in range(new_height):
        for i in range(new_width):
            x = float(width/new_width)*(i+0.5)-0.5
            y = float(heigh/new_height)*(j+0.5)-0.5

            # 3.1求出临近的四个点坐标
            x0 = int(x)
            x1 = min(x0 + 1, width-1)
            y0 = int(y)
            y1 = min(y0 + 1, heigh-1)
            # 3.2双线性插值
            empty_image[i,j] = (x1-x)*(y1-y)*img[x0,y0]+(x-x0)*(y1-y)*img[x1,y0]\
                               +(x1-x)*(y-y0)*img[x0,y1]+(x-x0)*(y-y0)*img[x1,y1]

    return empty_image


if __name__ == '__main__':

    img = cv2.imread("lenna.png")
    cv2.imshow("lod image", img)

    empty_image = bilinnear_inter(img,800,800)
    print(empty_image.shape)

    cv2.imshow("new image", empty_image)
    cv2.waitKey(0)

