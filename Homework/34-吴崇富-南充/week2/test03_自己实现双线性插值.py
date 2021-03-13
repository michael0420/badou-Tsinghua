#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

def bilinear_interpolation(img, out_dim):
    # 原始图像像素信息 (行对应高度,列对应宽度)
    src_h, src_w, channel = img.shape
    # 目标图像像素信息 (out_dim为(横向像素数,纵向像素数))
    dst_h, dst_w = out_dim[1], out_dim[0]
    print('src_h,src_w = ', src_h, src_w)
    print('dst_h,dst_w = ', dst_h, dst_w)
    # 原始图像与目标图像像素点数相同时的特例
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 原始图像相对于目标图像的缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # 新建空数组存储目标图像像素点
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    # 遍历目标图像每个通道每个像素点
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 目标像素点对应于原始图像的虚拟像素点
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 双线性差值公式中需要的邻近的像素点坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 应用双线性插值公式
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    # 返回目标图像
    return dst_img

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img = cv2.imread('lenna.png')
    dst_img = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interpolation', dst_img)
    cv2.imwrite('bilinear_interpolation_lenna.png',dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
