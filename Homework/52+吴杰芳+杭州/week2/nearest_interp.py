import cv2
import numpy as np

def nearest_interp(img, dst_h, dst_w):
    '''
    最邻近插值实现图像缩放
    :param img: 原图像矩阵，(h,w,c)
    :param dst_h: 缩放后图像高
    :param dst_w: 缩放后图像宽
    :return:
    '''
    H, W, C = img.shape
    print("src_img shape:", H, W)
    print("dst_img shape:", dst_h, dst_w)
    if H == dst_h and W == dst_h:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, C), dtype=np.uint8)
    x_scale = W / dst_w
    y_scale = H / dst_h
    for c in range(C):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 找出目的像素点对应的原图像上的虚拟点
                src_x = dst_x * x_scale
                src_y = dst_y * y_scale
                # 在原图像上找虚拟点的最邻近点作为目的像素点的值
                src_x0 = round(src_x)
                src_y0 = round(src_y)
                dst_img[dst_y, dst_x, c] = img[src_y0, src_x0, c]
    cv2.namedWindow("src img")
    cv2.namedWindow("dst img")
    cv2.imshow("src img", img)
    cv2.imshow("dst img", dst_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    nearest_interp(img , 700, 700)
