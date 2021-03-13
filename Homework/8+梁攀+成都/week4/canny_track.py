import cv2
import numpy as np

img = cv2.imread("cat.png")  #读取图片
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰度化

#track bar的回调函数实现
def canny_threshold_track(low_threshold):
    filter_img = cv2.GaussianBlur(gray, (3,3), 0, 0)  #高斯滤波，核为5，x方向标准差，y方向标准差
    edge_img = cv2.Canny(filter_img, low_threshold, low_threshold * 3, apertureSize = 3)  #apertureSize是soble算子的大小
    dist_img = cv2.bitwise_and(img, img, mask=edge_img) #用原始颜色添加到边缘上
    cv2.imshow("img edge",dist_img)

cv2.namedWindow("canny track", cv2.WINDOW_NORMAL) #创建一个窗口，大小可变

'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
low_threshold = 0
cv2.createTrackbar("阈值调节", "canny track", low_threshold, 200, canny_threshold_track)

canny_threshold_track(low_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
