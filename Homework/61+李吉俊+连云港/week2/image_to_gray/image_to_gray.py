
# 彩色图像的灰度化、二值化
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 灰度化（用cv的方法）
img = cv2.imread("F:/Small instance of algorithm/lenna.png")  # 读取图像
cv2.imshow("input", img)  # 显示img 图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
cv2.imshow("img_gray", img_gray)   # 显示img_gray 图像
cv2.waitKey(0)  # 等待键盘触发的时间，该值为零或负数时，表示无限等待。
cv2.destroyAllWindows()  # 用来释放（销毁）所有窗口
print(img_gray)  # 打印出图像的像素值

# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 128:
           img_gray[i, j] = 0
        else:
            img_gray[i, j] = 255
            
print("-----binary_image------")
print(img_gray)
print(img_gray.shape)
cv2.imshow("binary_image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 用 matplotlib 的方法对图像进行灰度化、二值化

'''plt.subplot(221)
src = plt.imread("F:/Small instance of algorithm/lenna.png")
plt.imshow(src)
print("----image lena----")
print(src)

img_gray = rgb2gray(src)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("----image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("----img_binary----")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()'''










