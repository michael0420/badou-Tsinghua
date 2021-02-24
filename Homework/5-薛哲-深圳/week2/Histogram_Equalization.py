#!/usr/bin/env python
# encoding=gbk

'''
ֱ��ͼ���⻯��cv2����cv2.equalization������ֱ�ӽ�ͼ�������⻯����
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

def Histogram(img):
    '''
    1.����ɨ��ԭʼͼ���ÿһ�����أ������ͼ��ĻҶ�ֱ��ͼH
    2.����Ҷ�ֱ��ͼ���ۼ�ֱ��ͼ
    :return:ԭͼ�ĻҶ�ֱ��ͼ
    '''
    h, w = img.shape
    hist0 = [0 for x in range(256)]
    for i in range(h):
        for j in range(w):
            hist0[img[i, j]] = hist0[img[i, j]] + 1
        pass
    return hist0

def Histogram_equalization(img, hist):
    '''
    �����ۼ�ֱ��ͼ��ֱ��ͼ���⻯ԭ��õ������������֮���ӳ���ϵ��ͼ����б任
    :param img: ԭͼ
    :param hist: ԭͼ���ۼ�ֱ��ͼ
    :return: ���⻯֮���ͼ��
    '''
    h, w = img.shape
    # equalization = [[0 for x in range(h)] for x in range(w)]
    # equalization = np.array(equalization)
    # print(type(equalization))
    for i in range(h):
        for j in range(w):
            sumPi = 0
            p = img[i, j]
            for k in range(p):
                sumPi = sumPi + hist[k]
                pass
            img[i, j] = int(sumPi*256/(h*w))-1
            sumPi = 0
            pass
        pass
    # np.array(equalization)
    # print(type(equalization))
    return img

# �Ҷ�ͼ��ľ��⻯
img = cv2.imread('lenna.png')
# print(type(img))

# hight, width, channels = img.shape
# channels = 1
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure('gray1') # ͼ�񴰿�����
plt.imshow(img)
plt.show()
# print(img)
# cv2.imshow('gray1', img)
# cv2.waitKey(0)

hist0 = Histogram(img)

equalization = Histogram_equalization(img, hist0)

plt.figure('gray') # ͼ�񴰿�����
plt.imshow(equalization)
plt.show()
cv2.imshow('gray', equalization)
cv2.imwrite('Hist_Equ_Graylenna.png', equalization)  # ����ͼ��
print(type(equalization))
print(equalization.shape)


# ��ɫͼ��ľ��⻯
img1 = cv2.imread('lenna.png')
hist0 = Histogram(img1[:, :, 0])
hist1 = Histogram(img1[:, :, 1])
hist2 = Histogram(img1[:, :, 2])

equalization0 = Histogram_equalization(img1[:, :, 0], hist0)
equalization1 = Histogram_equalization(img1[:, :, 1], hist1)
equalization2 = Histogram_equalization(img1[:, :, 2], hist2)
equalization = cv2.merge((equalization0, equalization1, equalization2))
equalization = cv2.cvtColor(equalization, cv2.COLOR_BGR2RGB)

cv2.imshow('gray', equalization)
cv2.imwrite('Hist_Equ_RGBlenna.png', equalization)  # ����ͼ��



# # �Ҷ�ͼ���ֱ��ͼ���⻯
# img = cv2.imread('lenna.png')  # ��ȡԭͼ
# gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)  # cv2Ĭ�϶�ȡBGR��ʽתΪgrayͼ
#
# # ֱ��ͼ���⻯
# dst = cv2.equalizeHist(gray)
# # ����ֱ��ͼ

