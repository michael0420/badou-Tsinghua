import cv2
import numpy as np


def hist_equalization(matrix):

    dict_pix = {}       # 字典是另一种可变容器模型，且可存储任意类型对象。
    dict_dst = {}       # 字典的每个键值 key=>value 对用冒号 : 分割，每个键值对之间用逗号 , 分割，整个字典包括在花括号 {} 中
    dst = np.zeros_like(matrix)    # 创建一个同matrix.shape的全0矩阵
    for i in range(len(matrix)):         #  len(matrix):返回二维数组中 列表的个数 ;用矩阵作为字典的键值
        for j in range(len(matrix[0])):  #  matrix[0]: 索引二维数组中第一个列表
            # 统计各键值出现总的次数
            dict_pix[matrix[i, j]] = dict_pix.get(matrix[i, j], 0)+1   # 字典get() 函数返回指定键的值，如果值不在字典中返回默认值。如果没有设定默认值，返回None;dict.get(key, default=None)
    print("----------------------------------------------------")
    print("dict_pix=",dict_pix)         # 打印出字典
    print("dict_keys=",dict_pix.keys())  # 打印出字典中全部的键值
    pixs = sorted(dict_pix.keys())        # 字典 keys()方法以列表返回一个字典所有的键; sorted() 函数对所有可迭代的对象进行排序操作。
    print("pixs=",pixs)
    print("----------------------------------------------------")
    print('pix\tNi\tPi=Ni/image\tsumPi\tsumPi*256-1\t四舍五入')          # \t:横向制表符，它的作用是对齐表格的各列。\n: 换行
    sumPi = 0
    for pix in pixs:               # 遍历列表元素，pixs是列表，pix为列表中元素
        print(pix, end='\t')       # end = “” 表示对象以什么结尾，默认是\n也就是换行
        print(dict_pix[pix], end='\t')
        image = matrix.shape[0]*matrix.shape[1]
        Pi = dict_pix[pix]/image
        print('%.2f\t\t' % Pi, end='')   #%字符：标记转换说明符
        sumPi += Pi
        print('%.2f\t' % sumPi, end='')
        print('%.2f\t\t' % (sumPi*256-1), end='')
        print(round(sumPi*256-1))
        dict_dst[pix] = round(sumPi*256-1)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            dst[i, j] = dict_dst[matrix[i, j]]
    print("----------------------------------------------------")
    print("dst=",dst)
    return dst


matrix = np.array([[1, 3, 9, 9, 8], [2, 1, 3, 7, 3], [3, 6, 0, 6, 4],[6, 8, 2, 0, 5], [2, 9, 2, 6, 0]])
hist_equalization(matrix)

img = cv2.imread("F:/Small instance of algorithm/esb.jpg")
src = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(src.shape)
hist_equalization(src)

