#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
    print(type(warpMatrix), warpMatrix.shape)

    # 之后为结果的后处理
    # warpMatrix = np.array(warpMatrix).T[0]# 转换为列向量，这里已经不需要这行代码了，因为warpMatrix已经是8行1列的矩阵(列向量)
    # warpMatrix = np.array(warpMatrix) #转换为列向量，这里已经不需要这行代码了，因为warpMatrix已经是8行1列的矩阵(列向量)
    # print(type(warpMatrix), warpMatrix.shape)
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0) # 插入a_33 = 1,插入后的索引为原warpMatrix.shape[0]
    warpMatrix = warpMatrix.reshape((3,3)) # 转换为3*3的形状
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0,457.0],[395.0,291.0],[624.0,291.0],[1000.0,457.0]] # 起始像素点坐标
    src = np.array(src)
    dst = [[46.0,920.0],[46.0,100.0],[600.0,100.0],[600.0,920.0]] # 目标像素点坐标
    dst = np.array(dst)
    warpMatrix = warpPerspectiveMatrix(src,dst)
    print(warpMatrix)
