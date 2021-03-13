import cv2
import numpy as np

img = cv2.imread("cat.png") #读取图片
img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度化
img_edge = cv2.Canny(img_gray, 150, 200)   #提取边缘
cv2.imshow("canny", img_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()