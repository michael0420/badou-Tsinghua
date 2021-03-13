import numpy as np
import cv2
from matplotlib import pyplot as plt


def calculate_grayhist(gray):
    '''
    统计灰度直方图
    :param gray: 灰度图矩阵，（h，w)
    :return:
    '''
    hist = np.zeros((256, 1))
    print(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            hist[gray[i,j]] += 1
    print("********hist******")
    print(hist)
    # plt.figure()
    # plt.subplot(121)
    # plt.hist(gray.ravel(), 256)
    # plt.subplot(122)
    # plt.plot(range(hist.shape[0]), hist)
    # plt.show()
    return hist

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    calculate_grayhist(gray)