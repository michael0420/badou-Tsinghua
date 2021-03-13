# -*- coding: utf-8 -*-

"""
@author: chengguo
Theme:双线性插值（放大或缩小）
"""

"""
坐标对应关系：
SrcX+0.5=(dstX+0.5)*(srcWidth/dstWidth)
SrcY+0.5=(dstY+0.5)*(srcHeight/dstHeight)
SrcX，SrcY是原始比例
dstX，dstY是目标比例
+0.5是为了让目标图像与原图像的几何中心点位置重合
"""

import numpy as np
import cv2


def bilinear_interp(img, out_img):
    src_h,src_w,channels=img.shape      #原始图像宽、高、通道数
    dst_h,dst_w=out_img[1],out_img[0]   #目标图像宽、高
    print("src_h,src_w",src_h,src_w)
    print("dst_h,dst_w",dst_h,dst_w)
    if src_h==dst_h and src_w==dst_w:   #比例相同则直接复制
        img.copy()
    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.uint8)        #定义目标图像
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h   #放缩比例
    for i in range(3):
        for dst_x in range(dst_w):
            for dst_y in range(dst_h):
                #套用上述公式
                SrcX=(dst_x+0.5)*scale_x-0.5
                SrcY=(dst_y+0.5)*scale_y-0.5

                #np.floor()函数用于以元素方式返回输入的下限
                src_x0=int(np.floor(SrcX))
                src_y0=int(np.floor(SrcY))
                src_x1=min(src_x0+1,src_w-1)
                src_y1=min(src_y0+1,src_h-1)

                temp0 = (src_x1 - SrcX) * img[src_y0,src_x0,i] + (SrcX - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - SrcX) * img[src_y1,src_x0,i] + (SrcX - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - SrcY) * temp0 + (SrcY - src_y0) * temp1)

    return dst_img



if __name__ == '__main__':
    img=cv2.imread("../res/lenna.png")
    dst=bilinear_interp(img,(800,800))   #传入原始图像和目标图像比例
    cv2.imshow("bilinear interp",dst)
    cv2.waitKey(0)