import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

def thresh_callback(val):
    threshold = val

    ## [Canny]
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    ## [Canny]

    ## [findContours]
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ## [findContours]

    ## [allthework]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    ## [allthework]

    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]

    ## [forContour]
    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])),
                     (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        print()
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    show_img('Contours', src)
    ## [showDrawings]

def show_img(name , img):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)

## [setup]
# Load source image
src = cv.imread('a01-014x.png')
# k = src.shape[0]//5
# src = src[k:src.shape[0]-k-100,:]
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


## [setup]
show_img('Source', src)
show_img('Source Gray', src_gray)
cv.imshow('Source', src)
thresh = 200 # initial threshold
thresh_callback(thresh)
## [trackbar]

cv.waitKey()