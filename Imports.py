from commonfunctions import *
import numpy as np
import numpy
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage.morphology import binary_erosion, binary_dilation
from skimage import feature
import cv2
import random
from skimage.exposure import histogram



