import cv2
import numpy as np


# 图像转换为灰度图
# 公式：Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
def image2gray(src, rWeight=0.2989, gWeight=0.5870, bWeight=0.1140):
    """图像转换为灰度图，与opencv效果一致

    :param src: 原始图像
    :param rWeight: R通道权重
    :param gWeight: G通道权重
    :param bWeight: B通道权重
    :return: 灰度图像
    """
    graySrc = None
    if src is not None:
        rgbArray = cv2.split(src)
        # (1)直接根据权重计算
        graySrc = (rgbArray[0] * bWeight + rgbArray[1] * gWeight + rgbArray[2] * rWeight).astype(np.uint8)
        # # (2)浮点运算转化为整数运算
        # bWeight = np.int32(bWeight * 10000)
        # gWeight = np.int32(gWeight * 10000)
        # rWeight = np.int32(rWeight * 10000)
        # graySrc = ((rgbArray[0].astype(np.int32) * bWeight +
        #             rgbArray[1].astype(np.int32) * gWeight +
        #             rgbArray[2].astype(np.int32) * rWeight)//10000).astype(np.uint8)
        # # (3)浮点运算转化为位移运算
        # offset = 8
        # bWeight = np.int32(np.round(bWeight * offset))
        # gWeight = np.int32(np.round(gWeight * offset))
        # rWeight = np.int32(np.round(offset - bWeight - gWeight))
        # graySrc = (rgbArray[0].astype(np.int32) * bWeight +
        #             rgbArray[1].astype(np.int32) * gWeight +
        #             rgbArray[2].astype(np.int32) * rWeight)
        # graySrc = (graySrc >> 3).astype(np.uint8)
    return graySrc


# 图像转换为灰度图(使用OpenCV接口)
def image2grayByOpencv(src):
    """图像转换为灰度图(使用OpenCV接口)

    :param src: 原始图像
    :return: 灰度图
    """
    graySrc = None
    if src is not None:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return graySrc