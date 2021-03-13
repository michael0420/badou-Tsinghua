import cv2
import numpy as np


def canny(img):
    if len(img) <= 0:
        print('img can not be empty')
        return
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaussian_image = gray_image
    # 增加高斯平滑处理
    gaussian_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    # Canny边缘检测
    edge_output = cv2.Canny(gaussian_image, 10, 150, apertureSize=3, L2gradient=True)

    # 在二值图像中寻找canny检测到的轮廓
    contours, hierarchy = cv2.findContours(edge_output.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    # 在原始图像填充多边型包围的区域
    # filled_image = cv2.fillPoly(img.copy(), np.array(contours), (0, 255, 0))

    contours_image = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), -1, hierarchy=hierarchy, maxLevel=1)

    cv2.imshow('contours_image', contours_image)
    # cv2.imshow('fill_on_src_image', filled_image)
    cv2.imshow('edge_output', edge_output)
    cv2.imshow('src', img)
    cv2.waitKey(0)


if __name__ == '__main__':

    img = cv2.imread('./lenna.png')
    canny(img)