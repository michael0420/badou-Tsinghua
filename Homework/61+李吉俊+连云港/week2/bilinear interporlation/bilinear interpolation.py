

import numpy as np
import cv2


# python implementation of bilinear interpolation
# python 实现双线性插值


def bilinear_interpolation(img, out_dim):      # 定义双线性插值函数 参数img为输入源图像；参数out_dim为目标图像的尺寸
    src_h, src_w, channel = img.shape          # 源图像的高、宽和通道数
    dst_h, dst_w = out_dim[1], out_dim[0]      # 目标图像的高和宽
    print("src_h, src_w = ", src_h, src_w)     # 打印出源图像的高和宽
    print("dst_h, dst_w = ", dst_h, dst_w)     # 打印出目标图像的高和宽
    if src_h == dst_h and src_w == dst_w:
        return img.copy()                      # 返回源图像的副本

    # 遍历目标图像，插值
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)   # 数据类型为 np.uint8，也就是0~255   #创建一个数组，通过往里面填值，形成新的图片

    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h    # scale_x为宽比例因子（源图像的宽与目标图像宽的比值）；scale_y为高比例因子（源图像的高与目标图像高的比值）
    for i in range(3):         # 对channel循环
        for dst_y in range(dst_h):     # 对height循环
            for dst_x in range(dst_w):   # 对width循环
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x

                # 目标图像在源图像上的坐标（从源图像上来看，它为我上面的一个虚拟点）
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                # 计算与虚拟点邻近的四个真实的像素点的位置（四个真实的像素点为源图像上的）；四个邻近点坐标，(src_x0,src_y0),(src_x0,src_y1),(src_x1,src_y0),(src_x1,src_y1)
                src_x0 = int(np.floor(src_x))         # 虚拟点取整后的横坐标
                src_y0 = int(np.floor(src_y))         # 从图像坐标上来看，四个真实点左上角点的坐标（src_x0，src_y0）；从数学计算的坐标系来看，四个真实点左下角点的坐标为（src_x0，src_y0）数学计算用这个坐标系下的坐标
                src_x1 = min(src_x0 + 1, src_w - 1)   # 与边界点比，取小的一个,减1是因为是从0开始算的
                src_y1 = min(src_y0 + 1, src_h - 1)   # 与边界点比，取小的一个,目的是防止超界

                # calculate the interpolation   计算双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)  # 插值计算后在取整

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("F:/Small instance of algorithm/esb.jpg")
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interpolation', dst)
    cv2.waitKey()


# “if __name__==’__main__:”也像是一个标志，象征着C/C++等语言中的程序主入口，告诉其他程序员，代码入口在此

# 对正实数向下取整：int()

# 四舍五入：round()

# 可以理解成向下取整：math.floor()

# 向上取整：math.ceil()


