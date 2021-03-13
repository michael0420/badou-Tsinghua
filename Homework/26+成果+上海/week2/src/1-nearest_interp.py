# -*- coding: utf-8 -*-

"""
@author: chengguo
Theme:最邻近插值法
"""
import numpy as np
import cv2

"""
返回来一个给定形状和类型的用0填充的数组
np.zeros(shape, dtype=float, order=‘C’)
shape:形状
dtype:数据类型，可选参数，默认numpy.float64
order:可选参数，c代表与c语言类似，行优先；F代表列优先
"""


def modify(img,newHeight,newWidth):
    height,width,channels=img.shape                                      #获取原生图片的宽、高、通道数
    newImage=np.zeros((newHeight,newHeight,channels),dtype=np.uint8)
    #进行映射
    sh=newHeight/height
    sw=newWidth/width
    for i in range(newHeight):
        for j in range(newWidth):
            x=int(i/sh)
            y=int(j/sw)
            newImage[i,j]=img[x,y]
    return newImage


if __name__ == '__main__':
    #step1、加载图片
    img=cv2.imread("../res/lenna.png")
    #step2、上采样
    zoom=modify(img,800,800)
    print(zoom.shape)
    cv2.imshow("nearest interp",zoom)
    cv2.imshow("image",img)
    cv2.waitKey(0)


"""
如果用户没有按下键,则继续等待 (循环)
常见 : 设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件
一般在 imgshow 的时候 , 如果设置 waitKey(0) , 代表按任意键继续
"""