# -*- coding: utf-8 -*-

import cv2
import numpy as np

def bilinear_interpolation(img, new_w,new_h):
    '''
    双线性插值实现图像缩放
    srcX + 0.5 = (destX + 0.5) * (srcWide / destWide)
    srcy + 0.5 = (destY + 0.5) * (srcHeight / destHeight)
    :param img: 原图像矩阵，（h, w, c)
    :param new_w: 缩放后图像宽
    :param new_h: 缩放后图像高
    :return:
    '''
    h, w, c = img.shape
    print("src_h, src_w = ", h, w)
    print("dst_h, dst_w = ", new_h, new_w)
    if h == new_h and w == new_w:
        return img.copy()
    x_scale = float(w) / new_w
    y_scale = float(h) / new_h
    dst_img = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for i in range(c):
        for dst_y in range(new_h):
            for dst_x in range(new_w):
                # 1. 找出目的像素点对应的原始坐标点(原图像上的虚拟点）
                srcX = (dst_x + 0.5) * x_scale - 0.5
                srcY = (dst_y + 0.5) * y_scale - 0.5
                # 2. srcX, srcY通常带小数，找出原图像上的虚拟点的邻近四个点
                src_x0 = int(np.floor(srcX)) # np.floor向下取整
                src_y0 = int(np.floor(srcY))
                src_x1 = min(src_x0 + 1, w - 1) # 防止超出原图像
                src_y1 = min(src_y0 + 1, h - 1)
                # 3.根据双线性插值公式求出目的像素坐标对应的像素值
                tmp0 = (src_x1 - srcX) * img[src_y0,src_x0,i] + (srcX - src_x0) * img[src_y0, src_x1, i]
                tmp1 = (src_x1 - srcX) * img[src_y1, src_x0,i] + (srcX - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - srcY) * tmp0 + (srcY - src_y0) * tmp1)
    return dst_img
if __name__ == '__main__':
    img = cv2.imread('./lenna.png')
    cv2.namedWindow("src img")
    dst_img = bilinear_interpolation(img, 700, 700)
    cv2.namedWindow("dst img")
    cv2.imshow("src img", img)
    cv2.imshow("dst img", dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

