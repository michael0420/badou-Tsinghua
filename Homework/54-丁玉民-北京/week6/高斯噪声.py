import cv2
import numpy as np
import random

def gaussian_noise(img, sigma, mean, percentage):

    h, w = np.shape(img)
    noise_img = img.copy()
    noise_num = h * w * percentage
    for i in range(int(noise_num)):

        randomX = random.randint(0, h - 1)
        randomY = random.randint(0, w - 1)
        noise_img[randomX][randomY] = noise_img[randomX][randomY] + random.gauss(mean, sigma)

        if noise_img[randomX][randomY] < 0:
            noise_img[randomX][randomY] = 0
        elif noise_img[randomX][randomY] > 255:
            noise_img[randomX][randomY] = 255

    return noise_img

def gaussian_rgb(img):
    noise_img = img.copy()
    b, g, r = cv2.split(noise_img)
    noise_b = gaussian_noise(b, 2, 4, 2)
    noise_g = gaussian_noise(g, 2, 4, 2)
    noise_r = gaussian_noise(r, 2, 4, 2)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imshow('noise_b', noise_b)
    cv2.imshow('noise_g', noise_g)
    cv2.imshow('noise_r', noise_r)
    noise_rgb = cv2.merge((noise_b, noise_g, r))
    cv2.imshow('noise_rgb', noise_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    img = cv2.imread('../data/lenna.png')
    #gaussian_rgb(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_img = gaussian_noise(img, 2, 4, 0.8)
    cv2.imshow('noise_img', noise_img)
    cv2.imshow('src', img)
    cv2.waitKey(0)