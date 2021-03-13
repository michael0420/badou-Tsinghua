#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pic_path = 'lenna.png'
# 1.灰度化
img = plt.imread(pic_path)
if pic_path[-4:] == '.png': # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img *= 255 # 还是浮点数类型
img = img.mean(axis=-1) # 三通道取均值实现灰度化
# img = img[:,:,0]*0.299+img[:,:,1]*0.587+img[:,:,2]*0.114 #实现灰度化的方式2
# img = cv2.imread(pic_path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 2.高斯平滑
sigma = 1.52 # 高斯平滑时的高斯核参数，标准差，可调
dim = int(np.round(6*sigma+1)) # round是四舍五入函数(距离最近的偶数)，根据标准差求高斯核是几乘几的，也就是维度
if dim % 2 == 0: # 最好是奇数，不是的话加一
    dim += 1
Gaussian_filter = np.zeros([dim,dim]) # 存储高斯核，这是数组不是列表了
tmp = [i-dim//2 for i in range(dim)] # 生成一个序列
# 计算高斯核
n1 = 1/(2*math.pi*sigma**2)
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter = Gaussian_filter/Gaussian_filter.sum() # 归到[0,1]之间
dx,dy = img.shape
img_new = np.zeros(img.shape) # 存储平滑之后的图像，zeros函数得到的是浮点型数据
tmp = dim//2 # 也可用(dim-1)//2
'''边缘填补 如果为constant模式，就得指定填补的值，如果不指定，则默认填充0。 
((1,1),(2,2))表示在二维数组array第一维（此处便是行）前面填充1行，最后面填充1行；
                 在二维数组array第二维（此处便是列）前面填充2列，最后面填充2列
如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
'''
img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter) # 注意img_new的i与img_pad的i不是同一行，后者相当于前者的i-dim//2
plt.figure(1)
plt.imshow(img_new.astype(np.uint8),cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off') # 关闭坐标刻度值

# 3、检测图像中的水平、垂直和对角边缘
sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # 水平梯度卷积核，检测垂直边缘
sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) # 垂直梯度卷积核，检测水平边缘sobel_kernel_y 是用上一排的像素减下一排的像素，这与图像y轴方向相反了，于是dy/dx时是用y轴向上的方向来判断
img_tidu_x = np.zeros(img_new.shape) # 存储梯度图像
img_tidu_y = np.zeros([dx,dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new,((1,1),(1,1)),'constant') # 边缘填补，根据sobel矩阵形状，所以写1
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x) # x方向
        img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y) # y方向
        img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001 # 方便后面做除法
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
plt.axis('off')

# 4. 非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1,dx-1):
    for j in range(1,dy-1):
        flag = True # 在8邻域内是否要保留做个标记
        temp = img_tidu[i-1:i+2,j-1:j+2] # 梯度幅值的8领域矩阵
        if angle[i,j] <= -1: # 使用线性插值法判断抑制与否
            num_1 = (temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1] # 画个示意图方便弄明白
            num_2 = (temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
            if not (img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                flag = False
        elif angle[i,j] >= 1:
            num_1 = (temp[0,2]-temp[0,1])/angle[i,j]+temp[0,1]
            num_2 = (temp[2,0]-temp[2,1])/angle[i,j]+temp[2,1]
            if not (img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                flag = False
        elif angle[i,j] > 0:
            num_1 = (temp[0,2]-temp[1,2])*angle[i,j]+temp[1,2]
            num_2 = (temp[2,0]-temp[1,0])*angle[i,j]+temp[1,0]
            if not (img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                flag = False
        elif angle[i,j] < 0:
            num_1 = (temp[1,0]-temp[0,0])*angle[i,j]+temp[1,0]
            num_2 = (temp[1,2]-temp[2,2])*angle[i,j]+temp[1,2]
            if not (img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                flag = False
        if flag:
            img_yizhi[i,j] = img_tidu[i,j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')
print(img_yizhi)

# 5、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = img_tidu.mean()*0.5
print(lower_boundary)
high_boundary = lower_boundary*3 # 这里设置高阈值是低阈值的三倍
print(high_boundary)
stack = []
for i in range(1,img_yizhi.shape[0]-1): # 外圈的梯度为0，不考虑了
    for j in range(1,img_yizhi.shape[1]-1):
        if img_yizhi[i,j] >= high_boundary: # 边缘像素的梯度值高于高阈值,标记为强边缘
            img_yizhi[i,j] = 255 # 置白
            stack.append([i,j])
        elif img_yizhi[i,j] <= lower_boundary: # 边缘像素的梯度值小于低阈值，则会被抑制。
            img_yizhi[i,j] = 0 # 置黑
# print(stack)
# 处理弱边缘像素
# 查看强边缘像素及其8个领域,只要其中一个为弱边缘像素，则该弱边缘像素点可以保留为真实的边缘。
while stack:
    x,y = stack.pop() # 出栈，得到一个强边缘像素点坐标
    # print(x,y)
    # 查看强边缘像素点8领域内是否有弱边缘像素，有则标记为强边缘像素点，并入栈
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if i!=x and j!=y:
                if lower_boundary < img_yizhi[i,j] < high_boundary:
                    img_yizhi[i,j] = 255 # 置白
                    stack.append([i,j]) # 像素点坐标入栈
# 不在强边缘像素点8领域内的弱边缘像素点置黑
img_yizhi[(img_yizhi!=0) & (img_yizhi!=255) ] = 0 # 使用布尔数组作为索引，简化代码,等价于如下代码
'''
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] != 0 and img_yizhi[i,j] != 255:
            img_yizhi[i,j] = 0
'''
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')
plt.show()






