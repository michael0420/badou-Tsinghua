import cv2
import numpy as np
import random
from skimage import util

def white_black_noise(img, percentage):

    h, w = np.shape(img)
    noise_img = img.copy()
    noise_num = h * w * percentage

    for i in range(int(noise_num)):

        randomX = random.randint(0, h - 1)
        randomY = random.randint(0, w - 1)

        noise_img[randomX][randomY] = random.choice([0, 255])

    return noise_img

def wb_noise_rgb(img):
    src_img = img.copy()
    b, g, r = cv2.split(src_img)
    noise_b = white_black_noise(b, 0.2)
    noise_g = white_black_noise(g, 0.2)
    noise_r = white_black_noise(r, 0.2)

    merge_img = cv2.merge((noise_b, noise_g, noise_r))

    cv2.imshow('noise_rgb', merge_img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def noise_util(img):
    '''
    mode:
    - 'gaussian'  Gaussian-distributed additive noise.
    - 'localvar'  Gaussian-distributed additive noise, with specified
                  local variance at each point of `image`.
    - 'poisson'   Poisson-distributed noise generated from the data.
    - 'salt'      Replaces random pixels with 1.
    - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                  -1 (for signed images).
    - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                  `low_val` is 0 for unsigned images or -1 for signed
                  images.
    - 'speckle'   Multiplicative noise using out = image + n*image, where
                  n is uniform noise with specified mean & variance.
    '''
    sp_noise_img = util.random_noise(img, mode='s&p', amount=0.2)
    cv2.imshow('sp_noise_img', sp_noise_img)
    cv2.imshow('src', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    img = cv2.imread('../data/lenna.png')
    wb_noise_rgb(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w_b_img = white_black_noise(img, 0.2)
    cv2.imshow('white black noise', w_b_img)
    cv2.imshow('src img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
