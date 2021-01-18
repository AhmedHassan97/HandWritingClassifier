from commonfunctions import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from skimage.transform import hough_line, hough_line_peaks


def detect_lines(image, min_angle=60, max_angle=90, start=-np.pi / 2, end=np.pi / 2, percision_total=360):
    coordinates = []
    if (len(image.shape) > 2):
        image = rgb2gray(image)
    edges = canny(image)
    tested_angles = np.linspace(start, end, percision_total)
    h, theta, d = hough_line(edges, theta=tested_angles)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    ax = axes.ravel()
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

        if angle >= (min_angle * np.pi) / 180 and angle <= (max_angle * np.pi) / 180:
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            coordinates.append(y0)
            coordinates.append(y1)
            ax[1].plot(origin, (y0, y1), '-r')

    ax[1].set_xlim(origin)
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('Detected lines')

    plt.tight_layout()
    plt.show()
    return coordinates