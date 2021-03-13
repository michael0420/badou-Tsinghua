import numpy as np
import cv2


def getWarpMatrix(src, dst):

    '''
    X' = a11x + a12y + a13 - a31xX' - a32X'y
    Y' = a21x + a22y + a23 - a31xY' - a32yY'
    根据公式求warpMatrix
    :param src: 源坐标点
    :param dst: 目标坐标点
    :return: warpMatrix
    '''
    nums = src.shape[0]
    A_m = np.zeros((2 * nums, 8))
    B_m = np.zeros((2 * nums, 1))

    for i in range(len(src)):
        A_i = src[i, :]
        B_i = dst[i, :]
        A_m[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B_m[2 * i] = B_i[0]

        A_m[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B_m[2 * i + 1] = B_i[1]

    A_m = np.mat(A_m)

    warpMatrix = A_m.I * B_m
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.append(warpMatrix, 1.0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def projection(img):
    img_copy = img.copy()

    # src = np.float32([[82, 53], [547, 61], [105, 400], [530, 411]])
    # dst = np.float32([[20, 20], [629, 20], [10, 480], [619, 480]])

    src = np.float32([[0, 0], [342, 0], [0, 342], [300, 300]])
    dst = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
    # warpMatrix = cv2.getPerspectiveTransform(src, dst)
    warpMatrix = getWarpMatrix(src, dst)
    # src: 原始图像
    # warpMatrix: 转换矩阵（3 * 3 transformation matrix）
    # dsize: 变换后的图像尺寸
    # flags: 插值方式，#INTER_LINEAR or #INTER_NEAREST
    # borderMode: 边界补偿方式，BORDER_CONSTANT or BORDER_REPLICATE
    # borderValue：边界补偿大小，常值，默认为0
    result = cv2.warpPerspective(img_copy, warpMatrix, dsize=(512,512), flags=2)
    cv2.imshow('src', img)
    cv2.imshow('result img', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('./new_image.png')
    projection(img)