import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])  #  图像的高 宽 中的最小值  350 420 ---》350 350
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    '''
    修改图片尺寸
    :param image: 图片列表
    :param size: 修改后的图片大小
    :return:
    '''
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    """
    文件转化为列表 并用下标数字对应分类类别，数字--类别==》下标==》类别
    :param argmax:
    :return:
    """
    with open("index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
    # print(synset[argmax])
    return synset[argmax]
