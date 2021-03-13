#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Canny边缘检测：优化的程序
'''
import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def CannyThreshold(lowThreshold):
    # 高斯滤波(优化项),为了降噪，可以做n次
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)  #参数:原图像,卷积核尺寸，标准差
    # canny检测边缘
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize=kernel_size) # sobel算子大小
    # 用原始颜色添加到检测的边缘上(用src1的颜色填到src2的mask上)
    '''
    cv2.bitwise_and()是对二进制数据进行“与”操作，
    是src1的颜色填到src2的mask上,用了mask取边缘检测后的图像后，边缘的像素点都置白(255)了，
    非边缘的像素点都置黑(0)了，这样在与原图像做二进制与操作的时候，原图像边缘的像素值保持不变，
    非边缘的像素值都是0。
    '''
    dst = cv2.bitwise_and(img,img,mask=detected_edges) # 参数:输入图像1、输入图像2、掩膜
    cv2.imshow('canny demo',dst) # 注意命名要跟图像窗口命名保持一致

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3


img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度化

cv2.namedWindow('canny demo') # 图像窗口命名,没有这一行代码不显示调节杠

'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
#设置调节杠,
cv2.createTrackbar('Min threshold','canny demo',lowThreshold,max_lowThreshold,CannyThreshold)

CannyThreshold(0) # 初始化，这一行代码必须要，否则不显示图像
if cv2.waitKey(0) == 27: # 等待按ESC结束cv2
    cv2.destroyAllWindows()


