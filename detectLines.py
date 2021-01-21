import cv2

from commonfunctions import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from skimage.transform import hough_line, hough_line_peaks


def detect_lines(binary):
    contours2,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    linesY=[]
    for contour in contours2:
        [x,y,w,h]=cv2.boundingRect(contour)
        if (binary.shape[1] / 2) < w < (binary.shape[1] * 5 / 6):
            linesY.append(y)
    linesY=np.sort(linesY)
    return linesY
