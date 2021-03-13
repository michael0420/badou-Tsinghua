import numpy as np
import cv2

def warp_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  #保证源和目标图像的样本至少是四个，这样就有8个点去计算8个未知数
    nums = src.shape[0]

    '''令a33为1，就只求8个系数，一个已知的坐标点可以列两个方程，所以乘以2
    矩阵A可以用已知的坐标点表示出，矩阵B也可以用已知的数据表示出'''
    A = np.zeros((nums * 2, 8))  #A初始化
    B = np.zeros((nums * 2, 1))  #B初始化

    #开始构造A矩阵和B矩阵
    for i in range(0, nums):   #一次循环算两行的系数
        srcx = src[i, 0]
        srcy = src[i, 1]
        dstx = dst[i, 0]
        dsty = dst[i, 1]
        A[i*2, :] = [srcx, srcy, 1, 0, 0, 0, -srcx*dstx, -srcy*dstx]  #一个坐标的第0行
        A[i*2+1, :] = [0, 0, 0, srcx, srcy, 1, -srcx*dsty, -srcy*dsty]
        B[i*2] = dstx
        B[i*2+1] = dsty
    A = np.mat(A) #生成矩阵
    #根据A*wrap=B => warp = A.I*B
    warp = A.I * B   #产生的是一个8行的列向量

    warp = np.array(warp).T[0] #转换成一个单纯的行向量，如果不加[0]则转换成一个1行8列的矩阵
    warp = np.insert(warp, warp.shape[0], values=1.0, axis=0) #在最后位置插入1
    warp = warp.reshape((3, 3)) #重新调整成一个3*3的矩阵
    return warp



if __name__ == "__main__":
    img = cv2.imread("cat.png")

    src = [[0, 0], [10, 0], [368, 372], [378, 372]]
    src = np.array(src)

    dst = [[0, 0], [20, 0], [358, 372], [378, 372]]
    dst = np.array(dst)

    warpMatrix = warp_matrix(src,dst)
    print(warpMatrix)

    src = np.float32([[0, 0], [10, 0], [368, 372], [378, 372]])
    dst = np.float32([[0, 0], [20, 0], [358, 372], [378, 372]])
    m = cv2.getPerspectiveTransform(src, dst)
    print(m)

    result = cv2.warpPerspective(img, warpMatrix, (538, 400))
    cv2.imshow("img", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)