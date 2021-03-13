#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import os

def pepper_salt_noise(src,percent):
    noise_num = int(percent*src.shape[0]*src.shape[1]) # 噪声点数目
    noise_img = src.copy()
    for i in range(noise_num):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，rand_x代表随机生成的行，rand_y代表随机生成的列
        # random.randint生成随机整数，生成的随机整数会有重复值，如果要去重的化，代码性能过低
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        if noise_img[rand_x,rand_y] != 0 and noise_img[rand_x,rand_y] != 255:
            # #random.random()生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
            if random.random() <= 0.5:
                noise_img[rand_x,rand_y] = 0
            else:
                noise_img[rand_x,rand_y] = 255
    return noise_img

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img1 = cv2.imread('lenna.png')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = pepper_salt_noise(img1,0.2)
    cv2.imshow('source',img1)
    cv2.imshow('lenna_pepper_salt_noise',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
