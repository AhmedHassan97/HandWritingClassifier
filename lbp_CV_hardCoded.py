from Imports import *


def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_center_pixel(img, x, y):
    center = img[x][y]

    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]
    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def lbp(img_gray):
    height, width = img_gray.shape
    # img_gray = cv2.cvtColor(img,
    #                         cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width),
                       np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_center_pixel(img_gray, i, j)
    # print(img_lbp)
    hestoImage = histogram(img_lbp, nbins=8)

    return hestoImage


def Euclidean_distance(v, u):
    return np.sqrt(np.sum((v - u) ** 2))


# path = 'texture_dotted.jpg'
# path2 = 'texture_dotted.jpg'
#
# img_bgr = cv2.imread(path, 1)
# img_bgr2 = cv2.imread(path2, 1)
#
# lbp1=lbp(img_bgr)
# lbp2=lbp(img_bgr)
#
# diff = Euclidean_distance(lbp1, lbp2)
# print(diff)
# print("LBP Program is finished")
