from Imports import *


def lbp_features(img):
    img_gray = rgb2gray(img)
    min_Image = np.min(img_gray)
    max_Image = np.max(img_gray)

    # img_gray = img_gray.astype(np.unit8)

    if max_Image - min_Image != 0:
        img_gray = (img_gray - min_Image) / (max_Image - min_Image)

    lbp = feature.local_binary_pattern(img_gray.astype(np.uint8), 8, 1, method="default")

    # lbp retorna um matriz com os códigos, então devemos extraír o histograma
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))

    # normaliza o histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    # return the histogram of Local Binary Patterns

    return hist


def Euclidean_distance(v, u):
    return np.sqrt(np.sum((v - u) ** 2))


# img=io.imread("a01-000u.png")
# io.imshow(img)
# io.show()


# img1 = io.imread("a.png")
# img2 = io.imread("a01-003.png")
#
# lbp1 = lbp_features(img1)
# lbp2 = lbp_features(img2)
#
# dQ2_H = Euclidean_distance(lbp1, lbp2)
# print(dQ2_H)

#
