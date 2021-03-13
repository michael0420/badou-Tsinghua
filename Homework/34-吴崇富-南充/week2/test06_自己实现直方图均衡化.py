#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from decimal import Decimal # 实现一般意义上的四舍五入

def equalizeHist(img):
    height,width,channel = img.shape
    img = img.copy()
    total = height*width # 像素点总数
    for k in range(channel):
        input_dict = dict() # 输入图像灰度和像素点坐标构成的字典
        sum_per = 0 # 不同灰度的像素点个数累和后所占比例
        output_dict = dict() # 输出图像灰度和像素点坐标构成的字典
        for i in range(height):
            for j in range(width):
                if img[i,j,k] not in input_dict:
                    input_dict[img[i,j,k]] = [(i,j,k)]
                else:
                    input_dict[img[i,j,k]].append((i,j,k))
        input_dict = dict(sorted(input_dict.items(),key=lambda x:x[0])) # 必须先排序，构造有序字典
        for key,value in input_dict.items():
            per = len(value)/total
            sum_per += per
            gray = int(Decimal(str(sum_per*256-1 if sum_per*256-1> 0 else 0)).quantize(Decimal('1.'),rounding='ROUND_HALF_UP'))
            if gray not in output_dict:
                output_dict[gray] = value
            else:
                output_dict[gray] += value

        for key,value in output_dict.items():
            for tup in value:
                img[tup[0],tup[1],tup[2]] = key
    return img

if __name__ == '__main__':
    # 切换当前工作路径为当前文件所在的父目录:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 这行代码在vscode里必须要加
    img = cv2.imread('lenna.png',1)
    # print(img.shape)
    dst = equalizeHist(img)
    cv2.imshow('rgb_histogram_equalization',np.hstack((img,dst)))
    cv2.waitKey(0)
