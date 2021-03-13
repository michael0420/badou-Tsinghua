import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def img2gray(imgpath):
    img = cv2.imread(imgpath)
    R,G,B = img[:,:,0],img[:,:,1],img[:,:,2]
    Gray = 0.299 * R + 0.587 * G + 0.114 * B   #图像灰度化
    return Gray

def gaussian_filter(img, sigma):
    dim = int(np.round(3 * sigma * 2 + 1))  #根据标准差计算高斯核的维度
    if dim % 2 == 0:    #确保高斯核的维度是基数
        dim += 1
    gaus_fil = np.zeros([dim, dim])  #初始化一个dim维的值为0的矩阵为高斯核
    #dim为3时，tmp=[-1 0 1]
    tmp = [i - dim//2 for i in range(dim)] #生成一个序列 //是除完之后向下取整
    #参数准备
    n1 = 1 / (2 * math.pi * sigma**2) #1/(2pi*sigma^2)
    n2 = -1/(2*sigma**2)   #-1/(2sigma^2)
    #计算高斯核的值
    for i in range(dim):
        for j in range(dim):
            #计算x和y两个方向的值，因为x-u和y-u,因此tmp已经把坐标对称了。
            gaus_fil[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))  #公式见ppt
    gaus_fil = gaus_fil / gaus_fil.sum()

    #开始用高斯核对对图像滤波
    imx, imy = img.shape
    img_new = np.zeros(img.shape) #初始化一个新的矩阵存储新图像
    #计算要填充多少列或行,为了把高斯核的中心对齐到图像边缘的点
    tmp = dim//2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant') #指定上下左右填充多少行，使用constant方式填充，默认值为0
    #对图像用高斯核进行滤波
    for i in range(imx):
        for j in range(imy):
            #从img_pad中取一个dim*dim的矩阵和卷积核相乘，然后值相加放到新的矩阵中
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*gaus_fil)
    return img_new

def soble_gradient(img):
    imx, imy = img.shape
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  #x方向梯度
    soble_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) #y方向梯度

    img_gradient_x = np.zeros(img.shape)  #存储求完x方向梯度的矩阵
    img_gradient_y = np.zeros(img.shape)  #存储求完y方向梯度的矩阵
    img_gradient = np.zeros(img.shape)  #存储梯度结果

    #由于上面用的卷积核是3维，因此只在周围填充一圈
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    #利用上面的卷积核求梯度幅值
    for i in range(imx):
        for j in range(imy):
            img_gradient_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x) #x方向
            img_gradient_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * soble_kernel_y) #y方向
            img_gradient[i ,j] = np.sqrt(img_gradient_x[i, j]**2 + img_gradient_y[i, j]**2)  #梯度幅值

    img_gradient_x[img_gradient_x == 0] = 0.00000001#为防止除0, 让为0的成为一个很小的非零的数
    angle = img_gradient_y / img_gradient_x  #梯度的tan值
    return  img_gradient, angle

def nms(img, angel):   #计算梯度方向的梯度值，比较像素的梯度值比这两个值大就留下，否则丢弃
    imx, imy = img.shape
    img_suppression = np.zeros([imx, imy])
    for i in range(1, imx-1):
        for j in range(1, imy-1):
            flag = True
            tmp = img[i-1:i+2, j-1:j+2]   #梯度幅值矩阵的8领域矩阵
            if angel[i, j] > -1 and angel[i, j] < 0: #左上靠近x轴和右下靠近x轴 ,在一根线上的比例运算
                gradient_tmp1 = (tmp[0, 0] - tmp[1, 0])*angel[i, j] + tmp[1, 0]
                gradient_tmp2 = (tmp[2, 2] - tmp[1, 2])*angel[i, j] + tmp[1, 2]
                if img[i, j] < gradient_tmp1 or img[i, j] < gradient_tmp2:
                    flag = False
            elif angel[i, j] <= -1:    #左上靠近y轴和右下靠近y轴,因为用到的夹角是90-angle,用除法
                gradient_tmp1 = (tmp[0, 0] - tmp[0, 1])/angel[i, j] + tmp[0, 1]
                gradient_tmp2 = (tmp[2, 2] - tmp[2, 1])/angel[i, j] + tmp[2, 1]
                if img[i, j] < gradient_tmp1 or img[i, j] < gradient_tmp2:
                    flag = False
            elif angel[i, j] >= 1:    #右上靠近y轴和左下靠近y轴,因为用到的夹角是90-angle,用除法
                gradient_tmp1 = (tmp[2, 0] - tmp[1, 0])/angel[i, j] + tmp[1, 0]
                gradient_tmp2 = (tmp[0, 2] - tmp[1, 2])/angel[i, j] + tmp[1, 2]
                if img[i, j] < gradient_tmp1 or img[i, j] < gradient_tmp2:
                    flag = False
            elif angel[i, j] < 1 and angel[i, j] > 0:  # 右上靠近x轴和左下靠近x轴
                gradient_tmp1 = angel[i, j] * (tmp[2, 0] - tmp[2, 1]) + tmp[2, 1]
                gradient_tmp2 = angel[i, j] * (tmp[0, 2] - tmp[0, 1]) + tmp[0, 1]
                if img[i, j] < gradient_tmp1 or img[i, j] < gradient_tmp2:
                    flag = False

            if flag:
                img_suppression[i, j] = img[i, j]
    return img_suppression

def dual_threshold_detec(img, low_boundary, high_boundary):   #双阈值检测和边缘连接
    imx, imy = img.shape
    stack = []
    #先把满足高阈值的点标记为边缘
    for i in range(1, imx - 1):
        for j in range(1, imy - 1):
            if img[i, j] >= high_boundary: #是边缘点
                img[i, j] = 255
                stack.append([i, j])   #把满足条件的点的坐标入栈
            elif img[i, j] <= low_boundary:
                img[i, j] = 0

    #接下来是边缘连接，判断周围的点是否和已经产生的边缘点挨着
    while len(stack) != 0:
        tmp_x, tmp_y = stack.pop()  #取出一个边缘点坐标
        #取出已经判断为边缘点的周围8个点
        matrix8 = img[tmp_x-1:tmp_x+2, tmp_y-1:tmp_y+2]
        if matrix8[0, 0]  < high_boundary and  matrix8[0, 0] > low_boundary:
            img[tmp_x - 1, tmp_y - 1] = 255 #标记为边缘
            stack.append([tmp_x - 1, tmp_y - 1])
        if matrix8[0, 1] < high_boundary and  matrix8[0, 1] > low_boundary:
            img[tmp_x - 1, tmp_y] = 255
            stack.append([tmp_x - 1, tmp_y])
        if matrix8[0, 2] < high_boundary and  matrix8[0, 2] > low_boundary:
            img[tmp_x - 1, tmp_y + 1] = 255
            stack.append([tmp_x - 1, tmp_y + 1])
        if matrix8[1, 0] < high_boundary and  matrix8[1, 0] > low_boundary:
            img[tmp_x, tmp_y - 1] = 255
            stack.append([tmp_x, tmp_y - 1])
        if matrix8[1, 2] < high_boundary and  matrix8[1, 2] > low_boundary:
            img[tmp_x, tmp_y + 1] = 255
            stack.append([tmp_x, tmp_y + 1])
        if matrix8[2, 0] < high_boundary and  matrix8[2, 0] > low_boundary:
            img[tmp_x + 1, tmp_y - 1] = 255
            stack.append([tmp_x + 1, tmp_y - 1])
        if matrix8[2, 1] < high_boundary and  matrix8[2, 1] > low_boundary:
            img[tmp_x + 1, tmp_y] = 255
            stack.append([tmp_x + 1, tmp_y])
        if matrix8[2, 2] < high_boundary and  matrix8[2, 2] > low_boundary:
            img[tmp_x + 1, tmp_y + 1] = 255
            stack.append([tmp_x + 1, tmp_y + 1])
    return img


img_gray = img2gray("cat.png")  #灰度化
plt.imshow(img_gray, cmap='gray')
plt.show()

img_fil = gaussian_filter(img_gray, 2.5)  #高斯滤波
plt.imshow(img_fil.astype(np.uint8), cmap='gray')
plt.show()

img_gradient, angel = soble_gradient(img_fil) #用sobel求梯度
plt.imshow(img_gradient.astype(np.uint8), cmap='gray')
plt.show()

img_suppression = nms(img_gradient, angel)  #对梯度幅值进行非最大值抑制
plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
plt.show()

img_edge = dual_threshold_detec(img_suppression, 18, 42) #双阈值检测和边缘连接
plt.imshow(img_edge.astype(np.uint8), cmap='gray')
plt.show()


