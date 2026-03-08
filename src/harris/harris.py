import cv2
import numpy as np


def harris_response(gray, k=0.04):

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    A = cv2.GaussianBlur(Ix2, (3,3), 1)
    B = cv2.GaussianBlur(Ixy, (3,3), 1)
    C = cv2.GaussianBlur(Iy2, (3,3), 1)

    det = A * C - B * B
    trace = A + C

    R = det - k * trace**2

    return R