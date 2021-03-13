import numpy as np
import cv2

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_PP_CENTERS

def kmeans(img, n_cluster):
    data = img.reshape((-1, 3))
    data = np.float32(data)

    retval, bestLabels, centers = cv2.kmeans(data, n_cluster, bestLabels=None, criteria=criteria, attempts=10, flags=flags)
    centers = np.uint8(centers)

    res = centers[bestLabels.flatten()]
    dst = res.reshape((img.shape))
    print(retval, bestLabels, centers)
    cv2.imshow('kmeans img', dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    print(cv2.__version__)
    img = cv2.imread('lenna.png')
    kmeans(img, 2)