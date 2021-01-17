import detectLines
from Imports import *
from detectLines import detect_lines

# read image
img = io.imread("a01-000u.png")
# print(img)
img = rgb2gray(img)
print(img.shape)

# ------------------------------------------------
# cropping
linePoints = detect_lines(img)
linePoints.append(int(img.shape[0] / 2))
linePoints = np.sort(linePoints)
indexOfCentre = np.where(linePoints == (int(img.shape[0] / 2)))
print(indexOfCentre[0])
indexBefore = indexOfCentre[0] - 1
indexAfter = indexOfCentre[0] + 1

cropped = img[int(linePoints[indexBefore]) + 2:int(linePoints[indexAfter]) - 2, 50:]
# cropped=img[1750+2:2000,30:]
# show_images([cropped, img], ["cropped", "original"])
# io.imshow(cropped)
# io.show()
# ------------------------------------------------


SE = np.ones((5, 200))
thr = threshold_otsu(cropped)
croppedBinarized = np.copy(cropped)
croppedBinarized[cropped < thr] = 0
croppedBinarized[cropped >= thr] = 1
dilatedImage = binary_dilation(croppedBinarized, SE)
closingImage = binary_erosion(croppedBinarized, SE)
# show_images([closingImage,croppedBinarized,cropped], ['Image After Closing','cropped Binarized','Cropped'])


# dividing cropped image into lines
img_gray = np.copy(closingImage)

count = 0
contours = find_contours(img_gray, 0.8)
bounding_boxes = []
for contour in contours:
    yMin = min(contour[:, 0])
    yMax = max(contour[:, 0])
    xMin = min(contour[:, 1])
    xMax = max(contour[:, 1])
    area = abs(xMax - xMin) * abs(yMax - yMin)
    # removing artifacts
    if area > 5000:
        bounding_boxes.append([int(xMin), int(xMax), int(yMin), int(yMax)])

img_with_boxes = np.copy(img_gray)

EachLineIndividually = []
EachLineIndividuallyBinarized = []
for box in bounding_boxes:
    [Xmin, Xmax, Ymin, Ymax] = box
    tempcoord = np.zeros((Ymax - Ymin, Xmax - Xmin))
    tempcoord = cropped[Ymin:Ymax, Xmin:Xmax]
    tempcoordBinarized = np.zeros((Ymax - Ymin, Xmax - Xmin))
    tempcoordBinarized = croppedBinarized[Ymin:Ymax, Xmin:Xmax]
    # rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=img_gray.shape)
    # show_images([tempcoord], ['Single line'])
    # show_images([tempcoordBinarized], ['Single line binarized'])
    tempcoordMultipliedByBinarized = tempcoord * (1 - tempcoordBinarized)
    tempcoordMultipliedByBinarized[tempcoordMultipliedByBinarized == 0] = 1
    # show_images([tempcoordMultipliedByBinarized], ['Single line binarized'])
    # tempcoord[rr, cc, count] = 1
    EachLineIndividually.append(tempcoordMultipliedByBinarized)
    EachLineIndividuallyBinarized.append(tempcoordBinarized)
    print(tempcoord.shape)
    count += 1
EachLineIndividually = np.asarray(EachLineIndividually)
# print(EachLineIndividually[6].shape)


# detecting connected components for each line
counter = 0
counterrr = 0
EachConnectedLineIndividually = []
avgHeights = []
maxHeightsSum = 0
Widths = []
for i in range(count):
    contours = find_contours(EachLineIndividuallyBinarized[i], 0.8)
    bounding_boxes = []
    maxHeight = 0
    avgHeight = 0
    width = 0
    for contour in contours:
        yMin = min(contour[:, 0])
        yMax = max(contour[:, 0])
        xMin = min(contour[:, 1])
        xMax = max(contour[:, 1])
        area = abs(xMax - xMin) * abs(yMax - yMin)
        # removing dots and commas
        if area > 1000:
            bounding_boxes.append([int(xMin), int(xMax), int(yMin), int(yMax)])
            width += int(Xmax - Xmin)
            if maxHeight < yMax - yMin:
                maxHeight = int(yMax - yMin)
            avgHeight = avgHeight + int(yMax - yMin)
    Widths.append(width)
    maxHeightsSum += maxHeight
    avgHeights.append(avgHeight // len(bounding_boxes))
    # print(area)
    # print(bounding_boxes)
    # print(len(bounding_boxes))
    img_with_boxes = np.copy(EachLineIndividually[i])
    ConnectedBlocks = np.ones(EachLineIndividually[i].shape) * 255

    previousX = 0
    Width = 0
    Height = maxHeight
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        rr, cc = rectangle(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=EachLineIndividually[i].shape)
        img_with_boxes[rr, cc] = 0

        Begining = abs(int((Ymax - Ymin) / 2) - int(Height / 2))
        ############################
        img_with_boxes_Inverted = 1 - img_with_boxes
        stored = img_with_boxes_Inverted * EachLineIndividually[i]
        # show_images([stored[Ymin:Ymax,Xmin:Xmax]], ['Lines with connected components and no horizontal spaces'])
        # print(stored[Ymin:Ymax,Xmin:Xmax].shape)
        # print(previousY,"previous Y")
        # print(Ymax-Ymin,"Diff")
        # print(Ymax-Ymin+previousY,"Add")
        # print(ConnectedBlocks[:,Xmin:Xmax].shape)
        ConnectedBlocks[Begining:Ymax - Ymin + Begining, previousX:Xmax - Xmin + previousX] = stored[Ymin:Ymax,
                                                                                              Xmin:Xmax]
        previousX += Xmax - Xmin
        Width = previousX
    ConnectedBlocks = ConnectedBlocks[0:Height, 0:Width]
    # show_images([stored[Ymin:Ymax,Xmin:Xmax]], ['Image With Bounding Boxes'])
    EachConnectedLineIndividually.append(ConnectedBlocks)

    ############################
    # show_images([1-img_with_boxes], ['Image With Bounding Boxes'])
    show_images([EachConnectedLineIndividually[i]], ['Lines with connected components and no horizontal spaces'])

# Super Texture Block Generation
superTextureBlock = np.ones((maxHeightsSum, np.max(Widths)))
print(superTextureBlock.shape)
yprevious = 150
for i in range(count):
    Begining2 = abs(int((EachConnectedLineIndividually[i].shape[0]) / 2) - int(yprevious / 2))
    superTextureBlock[Begining2:EachConnectedLineIndividually[i].shape[0] + Begining2,
    0:EachConnectedLineIndividually[i].shape[1]] *= EachConnectedLineIndividually[i]
    yprevious = yprevious + avgHeights[i] / 2

show_images([superTextureBlock], ['Lines with connected components and no horizontal spaces'])
