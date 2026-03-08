import cv2
import numpy as np


def nms(R, threshold_ratio=0.01):

    threshold = threshold_ratio * R.max()

    mask = R > threshold

    R_dilate = cv2.dilate(R, None)

    local_max = R == R_dilate

    corners = mask & local_max

    return corners