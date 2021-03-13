import cv2
import numpy as np


def nearestInterp(src, dstHeight, dstWidth):
    """
    最近邻插值

    :param src: 原始图像
    :param dstHeight: 输出图像高度
    :param dstWidth: 输出图像宽度
    :return: 输出图像
    """
    dst = None
    if src is not None:
        dst = np.zeros((dstHeight, dstWidth), dtype=src.dtype)

        srcHeight = np.float(src.shape[0])
        srcWidth = np.float(src.shape[1])

        rateY = srcHeight / dstHeight
        rateX = srcWidth / dstWidth

        for dstY in range(dstHeight):
            for dstX in range(dstWidth):
                # srcY = (dstY + 0.5) * rateY - 0.5  # Y方向中心位置对齐
                # srcX = (dstX + 0.5) * rateX - 0.5  # X方向中心位置对齐
                # srcY = max(min(srcY, dstHeight - 1), 0)
                # srcX = max(min(srcX, dstWidth - 1), 0)
                # srcIntY = int(srcY)
                # srcIntX = int(srcX)
                # srcCoodY = srcIntY if (srcY - srcIntY) < 0.5 else srcIntY + 1
                # srcCoodX = srcIntX if (srcX - srcIntX) < 0.5 else srcIntX + 1

                # 无中心对齐结果, 与opencv效果一致
                srcCoodX = int(dstX * rateX)
                srcCoodY = int(dstY * rateY)
                dst[dstY, dstX] = src[srcCoodY][srcCoodX]
    return dst


def bilinearInterp(src, dstH, dstW):
    """
    双线性插值

    :param src: 原始图像
    :param dstH: 输出图像高度
    :param dstW: 输出图像宽度
    :return: 输出图像
    """
    dst = None
    if src is not None:
        dst = np.zeros((dstH, dstW), dtype=np.uint8)
        srcH, srcW = src.shape
        scaleY = float(srcH) / dstH
        scaleX = float(srcW) / dstW
        for dstY in range(dstH):
            for dstX in range(dstW):
                srcY = (dstY + 0.5) * scaleY - 0.5  # Y方向中心位置对齐
                srcX = (dstX + 0.5) * scaleX - 0.5  # X方向中心位置对齐

                srcY1 = max(min(int(np.floor(srcY)), srcH - 1), 0)
                srcX1 = max(min(int(np.floor(srcX)), srcH - 1), 0)
                srcY2 = max(min(srcY1 + 1, srcH - 1), 0)
                srcX2 = max(min(srcX1 + 1, srcW - 1), 0)

                tempValueX2 = (srcX2 - srcX)  # / (srcX2 - srcX1)分母为1
                tempValueX1 = (srcX - srcX1)  # / (srcX2 - srcX1)分母为1
                dst[dstY, dstX] = int((srcY2 - srcY) * (tempValueX2 * src[srcY1, srcX1] + tempValueX1 * src[srcY1, srcX2]) + \
                                  (srcY - srcY1) * (tempValueX2 * src[srcY2, srcX1] + tempValueX1 * src[srcY2, srcX2]))
    return dst


def bilinear_interpolation(img, out_dim):
    """
    彩色图像双线性插值

    :param img: 原始图像
    :param out_dim: 输出纬度元组
    :return: 输出图像
    """
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


def interpByOpencv(src, dstHeight, dstWidth, interpFlag="INTER_NEAREST"):
    """
    图像插值(Opencv接口)
    :param src: 输入图像
    :param dstHeight: 输出图像高度
    :param dstWidth: 输出图像宽度
    :param interpFlag: 插值算法标签(与opencv的标签一致)
    :return: 输出图像
    """
    dst = None
    if src is not None:
        if interpFlag == "INTER_NEAREST":
            dst = cv2.resize(src, (dstHeight, dstWidth), interpolation=cv2.INTER_NEAREST)  # 最近邻插值
        elif interpFlag == "INTER_LINEAR":
            dst = cv2.resize(src, (dstHeight, dstWidth), interpolation=cv2.INTER_LINEAR)  # 双线性插值
        else:
            dst = cv2.resize(src, (dstHeight, dstWidth), interpolation=cv2.INTER_BITS2)  # ??
    return dst
