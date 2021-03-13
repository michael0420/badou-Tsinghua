#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def linear_test(base_img, dst_dim):
    base_h, base_w, channel = base_img.shape
    dst_h, dst_w = dst_dim[1], dst_dim[0]
    if base_h == dst_h and base_w == dst_w:
        return base_img
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(base_w) / dst_w, float(base_h) / dst_h

    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                base_x = (dst_x + 0.5) * scale_x - 0.5
                base_y = (dst_y + 0.5) * scale_y - 0.5

                base_x0 = int(np.floor(base_x))
                base_x1 = min(base_x0 + 1, base_w - 1)
                base_y0 = int(np.floor(base_y))
                base_y1 = min(base_y0 + 1, base_h - 1)
                FR1 = (base_x1 - base_x) * base_img[base_y0, base_x0, i] + (base_x - base_x0) * base_img[base_y0, base_x1, i]
                FR2 = (base_x1 - base_x) * base_img[base_y1, base_x0, i] + (base_x - base_x0) * base_img[base_y1, base_x1, i]
                dst_img[dst_y, dst_x, i] = int((base_y1 - base_y) * FR1 + (base_y - base_y0) * FR2)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = linear_test(img,(700,700))
    cv2.imshow('linear interp',dst)
    cv2.waitKey()