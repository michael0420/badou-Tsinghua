import image2gray
import interpolation
import histogram

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # Windows下plot标题乱码

if __name__ == '__main__':
    imagePath = 'lenna.png'
    # src = cv2.imread(imagePath, 0)  # 以灰度形式读取图像
    src = cv2.imread(imagePath)
    if src is not None:
        cv2.imshow('Src', src)

    # 图像转换为灰度图
    graySrc = image2gray.image2grayByOpencv(src)
    if graySrc is not None:
        cv2.imshow('GraySrcByOpencv', graySrc)
    graySrc = image2gray.image2gray(src)
    if graySrc is not None:
        cv2.imshow('GraySrc', graySrc)

    # # 最近邻插值
    # nearestInterpDst = interpolation.nearestInterp(graySrc, 800, 800)
    # if nearestInterpDst is not None:
    #     cv2.imshow('NearestInterpolation', nearestInterpDst)

    # 双线性插值
    bilinear2InterpDst = interpolation.interpByOpencv(graySrc, 800, 800, interpFlag='INTER_LINEAR')
    # bilinear2InterpDst = interpolation.bilinear_interpolation(graySrc, (800, 800))
    if bilinear2InterpDst is not None:
        cv2.imshow('Bilinear2Interpolation', bilinear2InterpDst)

    # # 灰度图像直方图可视化一
    # plt.figure()
    # plt.hist(graySrc.ravel(), 256, rwidth=1)
    # plt.show()

    # 灰度图像直方图可视化二
    hist = histogram.calcHistByOpencv(graySrc)
    fig = plt.figure()  # 新建一个图像
    plt.subplot(221), plt.title('灰度图', fontproperties=font_set), plt.imshow(graySrc, cmap='gray')
    plt.subplot(222)
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    hist = hist.astype(np.int)
    plt.bar(range(256), hist.ravel(), width=1)
    plt.xlim([0, 256])  # 设置x坐标轴范围
    # plt.show()

    # 直方图均衡化
    # equalizeDst = histogram.histEqualizationByOpencv(graySrc)
    equalizeDst = histogram.histEqualization(graySrc)
    if equalizeDst is not None:
        cv2.imshow("EqualizeDst", equalizeDst)
    plt.subplot(223), plt.title('灰度图均衡化', fontproperties=font_set), plt.imshow(equalizeDst, cmap='gray')
    hist = histogram.calcHistByOpencv(equalizeDst)
    # hist = histogram.calcHist(equalizeDst)
    plt.subplot(224)
    plt.title("Grayscale Histogram By Equalization")
    plt.xlabel("Bins")  # X轴标签
    plt.ylabel("# of Pixels")  # Y轴标签
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.bar(range(256), hist.ravel(), width=1)
    plt.show()

    # # 彩色图像直方图显示二
    # chans = cv2.split(src)
    # colors = ("b", "g", "r")
    # plt.figure()
    # plt.title("Flattened Color Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # for (chan, color) in zip(chans, colors):
    #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    #     plt.plot(hist, color=color)
    #     plt.xlim([0, 256])
    # plt.show()

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
