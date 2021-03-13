# -*- coding: utf-8 -*-

"""直方图均衡化"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1.获取灰度图像
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image", gray)

# 2.灰度图像均衡化
dst = cv2.equalizeHist(gray)
#cv2.imshow("equalization image", dst)

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()
