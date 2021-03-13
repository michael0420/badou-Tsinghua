#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import os

def gaussian_noise(src,mean,sigma,percent):
    noise_num = int(percent*src.shape[0]*src.shape[1]) # 噪声点数目
    noise_img = src.copy()
    for i in range(noise_num):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，rand_x代表随机生成的行，rand_y代表随机生成的列
        # random.randint生成随机整数，生成的随机整数会有重复值，如果要去重的化，代码性能过低
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        rand_channel = random.randint(0,src.shape[2]-1) # 任意一个通道的索引
        # 在原有像素灰度值上加上高斯随机数
        noise_img[rand_x,rand_y,rand_channel] = noise_img[rand_x,rand_y,rand_channel]+random.gauss(mean,sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255(注意!)
        if noise_img[rand_x,rand_y,rand_channel] < 0:
            noise_img[rand_x,rand_y,rand_channel] = 0
        if noise_img[rand_x,rand_y,rand_channel] > 255:
            noise_img[rand_x,rand_y,rand_channel] = 255
    return noise_img

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img1 = cv2.imread('lenna.png')
    img2 = gaussian_noise(img1,2,4,0.8)
    cv2.imshow('source',img1)
    cv2.imshow('lenna_gaussian_noise',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()