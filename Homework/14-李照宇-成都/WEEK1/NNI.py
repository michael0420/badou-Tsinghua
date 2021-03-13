# lzy_2.4
import cv2
import numpy as np
def function(img):
    width,height,channel = img.shape
    emptyimage = np.zeros((800, 800, channel), np.uint8) # 8 bit matrix
    for i in range(800):
        for j in range(800):
            x = int(i*width/800)
            y = int(j*height/800)  # not type of int
            emptyimage[i,j] = img[x,y]
    return emptyimage

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom.shape)
cv2.imshow("Origin",img)
cv2.imshow("800*800",zoom)
cv2.waitKey(0)