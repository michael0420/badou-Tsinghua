# -*- encoding=UTF-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def WarpPerspectiveMatrix(src,dst):
    # 定义应该要变换的长度
    nums = src.shape[0]  # 一般来说是4，因为图片是4个角，当然也可以增加到更多维度

    # 定义计算公式的A、B矩阵
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))

    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        # 代表着第0、2、4、6行
        A[i * 2] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2] = [B_i[0]]
        #采用i * 2 + 1，代表1、3、5、7行
        A[i * 2 + 1] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1] = [B_i[1]]
    #使用np.mat，是为了好拿到它的逆矩阵A.I
    A = np.mat(A)
    warpMatrix = A.I * B

    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape(3, 3)
    return  warpMatrix

def changeImg(warpMatrix, img1, tuple1):
    maxImgWdith, maxImgHeight = img1.shape[:2]
    zeroImg = np.zeros(tuple1, dtype=np.uint8)
    # 助教是将值取出，作简单计算
    # fix_matrix = warpMatrix
    # a_11 = fix_matrix[0, 0]
    # a_12 = fix_matrix[0, 1]
    # a_13 = fix_matrix[0, 2]
    # a_21 = fix_matrix[1, 0]
    # a_22 = fix_matrix[1, 1]
    # a_23 = fix_matrix[1, 2]
    # a_31 = fix_matrix[2, 0]
    # a_32 = fix_matrix[2, 1]
    # a_33 = fix_matrix[2, 2]
    warpMatrix = np.mat(warpMatrix).I
    for i in range(len(zeroImg)):
        for j in range(len(zeroImg[i])):
            orignal = np.dot(warpMatrix, np.array([i, j, 1]).T)
            z = orignal[0, 2]
            x = orignal[0, 0] / z
            y = orignal[0, 1] / z
            # 助教将原有公式作转换得到x,y的值并作简单计算，效率上高一些
            # Y_fix = j;
            # X_fix = i;
            # y_denominator = (a_31 * Y_fix - a_21) * a_12 - (a_31 * Y_fix - a_21) * a_32 * X_fix - (
            #             a_31 * X_fix - a_11) * a_22 + (a_31 * X_fix - a_11) * a_32 * Y_fix;
            # y = ((a_23 - a_33 * Y_fix - ((a_31 * Y_fix - a_21) * a_13) / (a_31 * X_fix - a_11)
            #       + ((a_31 * Y_fix - a_21) * a_33 * X_fix) / (a_31 * X_fix - a_11))
            #      * (a_31 * X_fix - a_11)) / y_denominator
            #
            # x = (a_12 * y + a_13 - (a_32 * y * X_fix + a_33 * X_fix)) / (a_31 * X_fix - a_11)

            x = int(np.round(x))
            y = int(np.round(y))
            # 超出位置作抛弃处理
            if x >= maxImgWdith:
                x = maxImgWdith - 1
            elif x <= 0:
                x = 0
            if y >= maxImgHeight:
                y = maxImgHeight - 1
            elif y <= 0:
                y = 0
            zeroImg[i, j] = img1[x, y]

    cv2.imshow("old", img1)
    cv2.imshow("new", zeroImg)

if __name__ == "__main__":
    #生成原始数据
    src = np.float32([[0,30],[ 512,30],[30,512],[512,512]])
    dst = np.float32([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])
    #dst = np.float32([[0,0],[ 512,0],[0,512],[512,512]])
    warpMatrix = WarpPerspectiveMatrix(src,dst)
    img1 = cv2.cvtColor(cv2.imread("images/lenna.png"), cv2.COLOR_BGR2GRAY)
    #changeImg(warpMatrix, img1, (512, 512))
    changeImg(warpMatrix, img1, (1024, 1024))
    cv2.waitKey(0)

