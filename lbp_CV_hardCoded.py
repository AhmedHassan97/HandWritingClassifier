from Imports import *


def get_pixel(img, center, x, y):
    new_value = 0

    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_center_pixel(img, x, y):
    center = img[x][y]

    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def lbp(img_gray):
    height, width = img_gray.shape
    img_lbp = np.zeros((height, width),
                       np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_center_pixel(img_gray, i, j)
    hestoImage = histogram(img_lbp, nbins=256)

    maximum = np.max(hestoImage)
    hestoImage = hestoImage[0] / maximum
    return hestoImage
