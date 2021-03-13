import numpy as np
import cv2

def bin_interpo(img, x, y):
    sh, sw, ch = img.shape
    dh = x
    dw = y
    if sh == dh and sw == dw:
        return img.copy()
    dst_img = np.zeros((dh, dw, ch), dtype=np.uint8)  #初始化一个矩阵
   #计算缩放比例
    scale_x = float(sw) / dw
    scale_y = float(sh) / dh

    for i in range(ch):    #通道遍历
        for dy in range(dh):  #竖轴遍历
            for dx in range(dw):   #横轴遍历
                #根据现有的坐标计算出在源图中要用到的虚拟坐标
                sx = (dx + 0.5) * scale_x - 0.5
                sy = (dy + 0.5) * scale_y - 0.5
                #计算原图中要用到的实际坐标
                src_x1 = int(np.floor(sx))
                src_x2 = min(src_x1 + 1, sw - 1)
                src_y1 = int(np.floor(sy))
                src_y2 = min(src_y1 + 1, sh - 1)

                #根据算法进行双线性插值
                tmp1 = (src_x2 - sx) * img[src_x1, src_y1, i] + (sx - src_x1) * img[src_x2, src_y1, i]
                tmp2 = (src_x2 - sx) * img[src_x1, src_y2, i] + (sx - src_x1) * img[src_x2, src_y2, i]

                tmp3 = (src_y2 - sy) * tmp1 + (sy - src_y1) * tmp2
                dst_img[dx, dy, i] = int(tmp3)

    return dst_img


img = cv2.imread('lenna.png')
dst = bin_interpo(img,200,200)
cv2.imshow('interp',dst)
cv2.waitKey(0)
