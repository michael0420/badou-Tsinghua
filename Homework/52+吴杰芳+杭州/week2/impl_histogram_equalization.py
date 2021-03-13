import numpy as np
import cv2
from matplotlib import pyplot as plt
from calculate_grayhist import calculate_grayhist

def impl_histogram_equalization(gray):
    '''
    灰度图直方图均衡化
    :param gray: 原灰度图像矩阵
    :return:
    '''
    H, W = gray.shape
    dst = np.zeros((H,W), dtype=np.uint8)
    # new_hist = np.zeros((256,1))
    # 原灰度级统计
    src_grayhist = calculate_grayhist(gray)
    beta = 256 / (H * W)
    count = 0
    for i in range(src_grayhist.shape[0]):
        Ni = src_grayhist[i]
        count += Ni
        dst[np.where(gray == i)] = max(0, count * beta - 1)
    plt.figure()
    plt.subplot(121)
    plt.hist(gray.ravel(), 256)
    plt.subplot(122)
    plt.hist(dst.ravel(), 256)
    plt.show()
    cv2.imshow("aa", np.hstack([gray,dst]))
    cv2.waitKey()

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    impl_histogram_equalization(gray)




