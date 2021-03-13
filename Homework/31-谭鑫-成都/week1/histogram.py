import cv2
import numpy as np


def calcHist(src):
    """
    图像直方图计算

    :param src: 输入图像
    :return: 输出图像
    """
    hist = None
    if src is not None:
        hist = np.zeros((256, 1), dtype=np.float)
        height, width = src.shape
        for y in range(height):
            for x in range(width):
                hist[src[y, x], 0] += 1
    return hist


def calcHistByOpencv(src, channel=0):
    """
    图像直方图计算(OpenCV接口)

    :param src: 输入图像
    :param channel: 通道索引
    :return: 输出图像
    """
    hist = None
    if src is not None:
        hist = cv2.calcHist([src], [channel], None, [256], [0, 256])
    return hist


def histEqualizationByOpencv(src):
    """
    直方图均衡化(Opencv接口)
    :param src: 原始图像
    :return: 输出图像
    """
    dst = None
    if src is not None:
        if len(src.shape) == 2:  # 灰度图
            dst = cv2.equalizeHist(src)
        else:  # 彩色图像
            (srcB, srcG, srcR) = cv2.split(src)
            dstB = cv2.equalizeHist(srcB)
            dstG = cv2.equalizeHist(srcG)
            dstR = cv2.equalizeHist(srcR)
            dst = cv2.merge((dstB, dstG, dstR))
    return dst


def grayEqualization(src, hist):
    """
    单通道图像直方图均衡

    :param src: 单通道图像(一般是灰度图)
    :param hist: 直方图(一维向量形式)
    :return: 均衡化结果
    """
    dst = None
    if len(hist.shape) == 1 and len(src.shape) == 2:
        dst = np.copy(src)
        src_h = src.shape[0]
        src_w = src.shape[1]
        # 计算直方图概率
        hist_rate = hist / src_h / src_w
        # 计算积分图
        hist_integral = np.zeros(256)
        hist_integral[0] = hist_rate[0]
        for i in range(1, 256):
            hist_integral[i] = hist_integral[i - 1] + hist_rate[i]
        # 计算灰度映射关系
        hist_map = (hist_integral * 256 - 1).astype(np.uint8)
        for i in range(256):
            check = src == i
            dst[check] = hist_map[i]
    return dst


def histEqualization(src):
    """
    直方图均衡化
    :param src: 原始图像
    :return: 输出图像
    """
    dst = None
    if src is not None:
        if len(src.shape) == 2:  # 灰度图
            hist = calcHist(src)
            dst = grayEqualization(src, hist.ravel())
        elif len(src.shape) == 3:  # 彩色图像
            (srcB, srcG, srcR) = cv2.split(src)
            histB = calcHist(srcB)
            histG = calcHist(srcG)
            histR = calcHist(srcR)
            dstB = grayEqualization(srcB, histB.ravel())
            dstG = grayEqualization(srcG, histG.ravel())
            dstR = grayEqualization(srcR, histR.ravel())
            dst = cv2.merge((dstB, dstG, dstR))
        else:
            dst = np.copy(src)
    return dst
