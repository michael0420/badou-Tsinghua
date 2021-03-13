import cv2
import numpy as np


def function(src):

    height, width, channels = src.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh)
            y = int(j/sw)  # 求目标图像像素的坐标（i，j）在源图像上的对应的坐标（取整后的结果）
            emptyImage[i, j] = src[x, y]

    return emptyImage

src = cv2.imread("F:/Small instance of algorithm/esb.jpg")

zoom = function(src)
print(zoom.shape)
cv2.imshow("image", src)
cv2.imshow("nearest interpolation", zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()