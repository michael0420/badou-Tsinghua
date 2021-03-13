# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

'''
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
lena = mpimg.imread('lena.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape#(512, 512, 3)
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
'''

'''
import cv2
imgcv = cv2.imread("./lenna.png")
img1 = cv2.imread("./img1.jpg")
cv2.imshow("imgcv",imgcv)
cv2.imshow("img 1",img1)
cv2.waitKey(0)
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def create_rgb_hist(image):
    """"创建 RGB 三通道直方图（直方图矩阵）"""
    h, w, c = image.shape
    print(f"h:{h},w:{w},c:{c}")
    # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
    # 16*16*16的意思为三通道每通道有16个bins
    rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
            index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
           	# 该处形成的矩阵即为直方图矩阵
            rgbhist[int(index), 0] += 1
    plt.ylim([0, 10000])
    plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)
    return rgbhist

def hist_compare(image1, image2):
    """直方图比较函数"""
    # 创建第一幅图的rgb三通道直方图（直方图矩阵）
    hist1 = create_rgb_hist(image1)
    # 创建第二幅图的rgb三通道直方图（直方图矩阵）
    hist2 = create_rgb_hist(image2)
    # 进行三种方式的直方图比较
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：%s, 相关性：%s, 卡方：%s" %(match1, match2, match3))


def image_zoom(image,beishu):
    """"创建 RGB 三通道直方图（直方图矩阵）"""
    h, w, c = image.shape
    h_dst = int(h*beishu)
    w_dst = int(w*beishu)
    #print(f"h:{h},w:{w},c:{c}  dst:h:{h_dst},w:{w_dst}")
    # 创建一个（height*height,3）的初始矩阵，作为直方图矩阵
    imgdst=np.zeros((h_dst,w_dst,3), np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            x_src = (col + 0.5) / beishu - 0.5
            y_src = (row + 0.5) / beishu - 0.5

            x1 = int(x_src)
            x2 = min(x1+1,w-1) #int(x_src + 1)
            y1 = int(y_src)
            y2 = min(y1+1,h-1) #int(y_src + 1)


            #print(f"src:{x_src} {y_src}  xxyy:{x1} {x2} {y1} {y2}")
            #b = image[x_src, y_src, 0]
            #g = image[x_src, y_src, 1]
            #r = image[x_src, y_src, 2]

            b_dst = (y2 - y_src) * ((x2 - x_src) * image[y1, x1, 0] + (x_src - x1) * image[y1, x2, 0]) + \
                    (y_src - y1) * ((x2 - x_src) * image[y2, x1, 0] + (x_src - x1) * image[y2, x2, 0] )

            g_dst = (y2 - y_src) * ((x2 - x_src) * image[y1, x1, 1] + (x_src - x1) * image[y1, x2, 1]) + \
                    (y_src - y1) * ((x2 - x_src) * image[y2, x1, 1] + (x_src - x1) * image[y2, x2, 1])

            r_dst = (y2 - y_src) * ((x2 - x_src) * image[y1, x1, 2] + (x_src - x1) * image[y1, x2, 2]) + \
                    (y_src - y1) * ((x2 - x_src) * image[y2, x1, 2] + (x_src - x1) * image[y2, x2, 2])
            # 赋值到图像数组
            imgdst[row, col, 0] = b_dst
            imgdst[row, col, 1] = g_dst
            imgdst[row, col, 2] = r_dst
    cv.imwrite('./zoom.png',imgdst)
    r, g, b = cv.split(imgdst)
    return cv.merge([r,g,b])


##计算bgr直方图
def img_hist(image):
    """"创建 RGB 三通道直方图（直方图矩阵）"""
    h, w, c = image.shape
    imghist = np.zeros((256, 3), np.uint32)
    a = np.arange(0, 255)#= [0] * 256

    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]

            imghist[image[row, col, 0], 0] +=1
            imghist[image[row, col, 1], 1] +=1
            imghist[image[row, col, 2], 2] +=1

    plt.subplot(1, 1, 1)
    plt.title("hist")
    plt.plot(imghist[:, 0], 'b', imghist[:, 1], 'g', imghist[:, 2],'r')
    plt.show()
    return imghist


def img_hist_equalization(image):
    h, w, c = image.shape
    imghistf = np.zeros((256, 3), np.float)
    imghistequ = np.zeros((256, 3), np.uint8)
    imghist = img_hist(image)
    imgdst = np.zeros((h, w, 3), np.uint8)
    for i in range(256):
        if i>0:
            imghist[i, 0] += imghist[i-1, 0]
            imghist[i, 1] += imghist[i-1, 1]
            imghist[i, 2] += imghist[i-1, 2]
        imghistf[i,0] = imghist[i,0]/(h*w)
        imghistf[i,1] = imghist[i,1]/(h*w)
        imghistf[i,2] = imghist[i,2]/(h*w)

    for i in range(256):
        imghistequ[i,0] = imghistf[i,0]*255
        imghistequ[i,1] = imghistf[i,1]*255
        imghistequ[i,2] = imghistf[i,2]*255

    for row in range(h):
        for col in range(w):
            imgdst[row, col, 0] = imghistequ[image[row, col, 0],0]
            imgdst[row, col, 1] = imghistequ[image[row, col, 1],1]
            imgdst[row, col, 2] = imghistequ[image[row, col, 2],2]

    cv.imwrite('./imgequ.png',imgdst)
    r, g, b = cv.split(imgdst)
    return cv.merge([r,g,b])

##计算yuv  y直方图
def imgyuv_hist(image):
    """"创建 YUV 三通道直方图（Y直方图矩阵）"""
    h, w, c = image.shape
    imghist = np.zeros(256, np.uint32)
    a = np.arange(0, 255)#= [0] * 256
    for row in range(h):
        for col in range(w):
            y = image[row, col, 0]
            imghist[image[row, col, 0]] +=1
    plt.subplot(1, 1, 1)
    plt.title("hist")
    plt.plot(imghist, 'r')
    plt.show()
    return imghist


def imgyuv_hist_equ(image):
    h, w, c = image.shape
    print(image.shape)
    imghistf = np.zeros(256, np.float)
    imghistequ = np.zeros(256, np.uint8)
    imghist = imgyuv_hist(image)
    imgdst = np.zeros((h, w, 3), np.uint8)
    for i in range(256):
        if i > 0:
            imghist[i] += imghist[i - 1]
        imghistf[i] = imghist[i] / (h * w)

    for i in range(256):
        imghistequ[i] = imghistf[i] * 255

    for row in range(h):
        for col in range(w):
            imgdst[row, col, 0] = imghistequ[image[row, col, 0]]
            #imgdst[row, col, 0] = image[row, col, 0]#imghistequ[image[row, col, 0]]
            imgdst[row, col, 1] = image[row, col, 1]
            imgdst[row, col, 2] = image[row, col, 2]

    cv.imwrite('./imgequyuv.png', imgdst)
    y, cb, cr = cv.split(imgdst)
    imgyuv = cv.merge([y, cb, cr])

    imghisty = imgyuv_hist(imgyuv)
    return cv.cvtColor(imgyuv,cv.COLOR_YCrCb2BGR)






src1 = cv.imread("../../../../img/8.jpg")
cv.imshow("diff1", src1)


imgz = image_zoom(src1,0.3)
imgequ = img_hist_equalization(imgz)
cv.imshow("zoomimg equ", imgequ)
imgyuv = cv.cvtColor(imgz,cv.COLOR_BGR2YCrCb)
imgyuvequ=imgyuv_hist_equ(imgyuv)
cv.imshow("zoomimg imgyuv equ", imgyuvequ)


#直方图均衡库调用,用于对比
(b, g, r) = cv.split(src1)
bH = cv.equalizeHist(b)
gH = cv.equalizeHist(g)
rH = cv.equalizeHist(r)
# 合并每一个通道
result = cv.merge((bH, gH, rH))
cv.imshow("dst_rgb", result)




cv.waitKey(0)
cv.destroyAllWindows()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
