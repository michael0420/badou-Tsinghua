#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def warpPerspectiveMatrix(src,dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 # 断言成立则执行下面的代码，断言不成立则报错
    nums = src.shape[0]
    A = np.zeros((2*nums,8)) # A*warpMatrix=B
    B = np.zeros((2*nums,1))
    for i in range(nums): # 遍历每个起始像素点和目标像素点坐标的索引
        A_i = src[i,:] # 起始像素点坐标
        B_i = dst[i,:] # 目标像素点坐标
        # 找规律给A和B赋值
        A[2*i,:] = [A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2*i+1,:] = [0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]

    A = np.mat(A) # 将A转换为矩阵
    # 用A.I或A**-1求出矩阵A的逆矩阵，然后与B相乘，求出warpMatrix
    # warpMatrix = A.I*B # #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warpMatrix = A**-1*B
    #求逆矩阵的另外一种方式: np.linalg.inv()
    # warpMatrix = np.linalg.inv(A).dot(B)
    # print(type(warpMatrix), warpMatrix.shape)
    # 之后为结果的后处理
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0) # 插入a_33 = 1,插入后的索引为原warpMatrix.shape[0]
    warpMatrix = warpMatrix.reshape((3,3)) # 转换为3*3的形状
    return warpMatrix

def warpPerspective(src_img,warpMatrix,dsize): # 传入原始图像、warp矩阵、图像尺寸(宽度，高度)
    '''
    从目标像素点坐标推出原图像像素点坐标(如果从原图像像素点坐标去推目标像素点坐标，会引入噪声点，因为有不需要的点转化过去)
    '''
    src_height,src_width,channel = src_img.shape
    dst_width,dst_height = dsize
    dst_img = np.zeros((dst_height+1,dst_width+1,channel)) # 因为像素点坐标包含等于dst_height,dst_width的情况
    # 不需要对每个通道都遍历，因为只是找像素点坐标的对应关系，像素值不参与计算
    for dst_y in range(dst_height+1):
        for dst_x in range(dst_width+1):
            # np.expand_dims()扩维,必须传axis参数，axis=1表示行方向
            # src_x_y_z = warpMatrix.I*np.mat(np.expand_dims([dst_x,dst_y,1],axis=1)) # 求得的矩阵带深度信息
            # 上行代码等价于下面行代码
            # src_x_y_z = warpMatrix.I*np.mat(np.array([dst_x,dst_y,1]).reshape((-1,1)))
            # 上行代码也等价于下面行代码
            src_x_y_z = warpMatrix.I*np.mat(f'{dst_x};{dst_y};1') # 求得的矩阵带深度信息
            src_x,src_y,src_z = src_x_y_z[0],src_x_y_z[1],src_x_y_z[2]
            src_x = int(src_x/src_z)
            src_y = int(src_y/src_z)
            if 0<=src_x<src_width and 0<=src_y<src_height:
                dst_img[dst_y,dst_x] = src_img[src_y,src_x] # 三通道图像赋的是对应像素坐标三通道的像素值
    dst_img = dst_img.astype(np.uint8) # 这行代码要加上，否则可能格式不对，因为warpMatrix中可能含小数
    return dst_img
'''
# 参考的是下面代码
def myWarpPerspective(warp_img,warp_matrix,goal_shape):
    goal_x, goal_y = goal_shape
    channel = warp_img.shape[2]
    # for c in range(warp_img.shape):
    fix_matrix = warp_matrix
    fix_img = np.zeros((goal_y, goal_x, channel))

    max_y = warp_img.shape[0] - 1
    max_x = warp_img.shape[1] - 1
    for Y_fix in range(goal_y):  # height or y
        for X_fix in range(goal_x):  # width or x
            # X is dst axis, the same as Y; x is orig axis,so y is same;
            # formula: [[X],[Y],[Z]] = fix_matrix * [[x],[y],[1]]
            # => fix_matrix.I * [[X],[Y],[Z]] = [[x],[y],[1]]
            # => [[x],[y],[1]] = fix_matrix.I * [[X],[Y],[Z]] # but we don't know Z, so we replace Z with 1 ;
            # => [[x],[y],[?]] = fix_matrix.I * [[X],[Y],[1]]
            # => [[x],[y],[z]] = fix_matrix.I * [[X],[Y],[1]]
            # x' = x/z; and y' = y/z
            orig_x_y = fix_matrix.I * np.mat(np.expand_dims([X_fix, Y_fix, 1], axis=1))
            x = orig_x_y[0][0]
            y = orig_x_y[1][0]
            z = orig_x_y[2][0]  # Note: z != 1

            y = int(np.round(y / z))
            x = int(np.round(x / z))
            if y < 0:
                y = 0
            if y > max_y:
                y = max_y
            if x < 0:
                x = 0
            if x > max_x:
                x = max_x

            fix_img[Y_fix, X_fix] = warp_img[y, x]
            # print("X_fix:",X_fix,"Y_fix:",Y_fix,",x:",x,",y:",y)
            fix_img = fix_img.astype(np.uint8)
    return fix_img
'''

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    img = cv2.imread('photo1.jpg')
    result3 = img.copy()
    # img = cv2.GaussianBlur(img,(3,3),0) # 高斯滤波,用来降噪
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度化，边缘检测要求输入灰度图像
    # edges= cv2.Canny(gray,50,150,apertureSize=3) # Canny检测边缘，方便找用于透视变换的顶点
    # cv2.imwrite('canny.jpg',edges) # 保存图像
    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32([[207,151],[517,285],[17,601],[343,731]]) # 原始图像的4个顶点坐标，可以用画图工具或matplotlib来找
    # import math
    # width = int(math.sqrt((207-517)**2+(151-285)**2)) # 337
    # height = int(math.sqrt((207-17)**2+(151-601)**2)) # 488
    # print(f'变换后图像的高度:{height},宽度:{width}')
    dst = np.float32([[0,0],[337,0],[0,488],[337,488]]) # 目标图像的4个顶点坐标
    # 生成透视变换矩阵，进行透视变换
    m = warpPerspectiveMatrix(src,dst)
    # result = myWarpPerspective(result3,m,(337,488))
    result = warpPerspective(result3,m,(337,488))
    cv2.imshow('src',img)
    cv2.imshow('result',result)
    cv2.waitKey(0)