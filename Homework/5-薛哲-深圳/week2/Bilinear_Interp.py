import cv2
import numpy as np
'''
双线性插值
'''


def bilinear_interpolation(img, out_dim):
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
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 使插值前后的图像几何中心重合防止插值后的图像偏左上

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))  # 向下取整得到y不变时第一个x坐标
                src_x1 = min(src_x0 + 1, src_w - 1)  # x0变化一个坐标值后的坐标
                src_y0 = int(np.floor(src_y))  # 向下取整得到x不变时第一个y坐标
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 两两组合可得四个点

                # calculate the interpolation y不变，选两点，在x方向上做插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                # 利用x方向上做插值得到的两点像素值再在y方向上做插值得到最后的像素值
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imwrite('bilinearInterplenna.png', dst)  # 保存插值后的图
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()