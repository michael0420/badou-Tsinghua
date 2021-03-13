import cv2
import numpy as np  # 导包
'''
最邻近插值
'''

def function(img):
    height, width, channels = img.shape  # 原图像的高，宽，颜色通道
    emptyImage = np.zeros((800, 800, channels), np.uint8)  # 通过插值得到800*800的新图像
    sh = 800/height
    sw = 800/width  # 新图与原图的尺寸比例
    for i in range(800):
        for j in range(800):
            x = round(i/sh)
            y = round(j/sw)
            # 遍历新图每个像素点，求出该点对应原图上的像素点坐标
            emptyImage[i][j] = img[x][y]  # 将原图像素值赋给新图
    return emptyImage

img = cv2.imread('lenna.png')
zoom = function(img)
cv2.imwrite('nearestInterplenna.png', zoom)  # 保存插值后的图
print(zoom.shape)
cv2.imshow("nearest interp", zoom)  # 展示新图
cv2.imshow("lenna", img)  # 展示原图

cv2.waitKey(0)

