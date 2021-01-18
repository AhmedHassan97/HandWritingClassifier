from Imports import *
from detectLines import detect_lines


def GetTextureBlock(imageName):
    # read image
    img = io.imread("./images/a02-050.png")
    img = rgb2gray(img)

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

    # ------------------------------------------------

    thr = threshold_otsu(cropped)
    croppedBinarized = np.copy(cropped)
    croppedBinarized[cropped < thr] = 0
    croppedBinarized[cropped >= thr] = 1
    croppedBinarized = 1 - croppedBinarized
    kernel = np.ones((1, 150), np.uint8)
    dilatedImage = cv2.dilate(croppedBinarized, kernel, iterations=1)
    closingImage = cv2.erode(dilatedImage, kernel, iterations=1)
    croppedBinarized = 1 - croppedBinarized
    closingImage = 1 - closingImage
    # show_images([closingImage, croppedBinarized, cropped], ['Image After Closing', 'cropped Binarized', 'Cropped'])

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
        if area > 10000:
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

        tempcoordMultipliedByBinarized = tempcoord * (1 - tempcoordBinarized)
        # show_images([tempcoordMultipliedByBinarized], ['Single line binarized'])
        if tempcoord.shape[0] * tempcoord.shape[1] > 60000:
            EachLineIndividually.append(tempcoordMultipliedByBinarized)
            EachLineIndividuallyBinarized.append(tempcoordBinarized)
            print(tempcoord.shape)
            count += 1
    EachLineIndividually = np.asarray(EachLineIndividually)
    # for i in range(count):
        # show_images([EachLineIndividually[i]], ['Single line'])

    # detecting connected components for each line and centering them and removing horizontal spaces
    countNew = 0
    EachConnectedLineIndividually = []
    avgHeights = []
    maxHeightsSum = 0
    WidthsforLines = []
    for i in range(count):
        contours = find_contours(EachLineIndividuallyBinarized[i], 0.8)
        bounding_boxes = []
        maxHeightofLine = 0
        avgHeight = 0
        sumofWidthsofEachLine = 0
        countForAvgheight = 0
        for contour in contours:
            yMin = min(contour[:, 0])
            yMax = max(contour[:, 0])
            xMin = min(contour[:, 1])
            xMax = max(contour[:, 1])
            area = abs(xMax - xMin) * abs(yMax - yMin)
            # removing dots and commas
            if area > 1000:
                bounding_boxes.append([int(xMin), int(xMax), int(yMin), int(yMax)])

        img_with_boxes = np.copy(EachLineIndividually[i])
        ConnectedBlocks = np.zeros(EachLineIndividually[i].shape)

        previousX = 0
        Width = 0
        for box in bounding_boxes:
            [Xmin, Xmax, Ymin, Ymax] = box
            rr, cc = rectangle(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=EachLineIndividually[i].shape)
            img_with_boxes[rr, cc] = 0

            ############################ Getting heights and widths
            if maxHeightofLine < Ymax - Ymin:
                maxHeightofLine = Ymax - Ymin
            avgHeight += Ymax - Ymin
            countForAvgheight += 1
            sumofWidthsofEachLine += Xmax - Xmin
            #############################

            Begining = abs(int((Ymax - Ymin) / 2) - int(maxHeightofLine / 2))
            img_with_boxes_Inverted = 1 - img_with_boxes
            stored = img_with_boxes_Inverted * EachLineIndividually[i]
            #################################if more than one line is detected
            if Xmax - Xmin + previousX > ConnectedBlocks.shape[1]:
                print("aaaaaaaaaa")
                ConnectedBlocks = ConnectedBlocks[0:maxHeightofLine, 0:Width]
                EachConnectedLineIndividually.append(ConnectedBlocks)
                countNew += 1
                ConnectedBlocks = np.zeros(EachLineIndividually[i].shape)
                previousX = 0
                maxHeightsSum += maxHeightofLine
                avgHeights.append(avgHeight // countForAvgheight)
                WidthsforLines.append(sumofWidthsofEachLine)
                sumofWidthsofEachLine = 0
                maxHeightofLine = 0
                avgHeight = 0
                countForAvgheight = 0
            #################################
            ConnectedBlocks[Begining:Ymax - Ymin + Begining, previousX:Xmax - Xmin + previousX] = stored[Ymin:Ymax,
                                                                                                  Xmin:Xmax]
            previousX += Xmax - Xmin
            Width = previousX
        ConnectedBlocks = ConnectedBlocks[0:maxHeightofLine, 0:Width]
        if ConnectedBlocks.shape[0] * ConnectedBlocks.shape[1] > 60000:
            EachConnectedLineIndividually.append(ConnectedBlocks)
            countNew += 1
            maxHeightsSum += maxHeightofLine
            avgHeights.append(avgHeight / int(countForAvgheight + 1))
            WidthsforLines.append(sumofWidthsofEachLine)
    print(countNew, count)
    for i in range(countNew):
        # show_images([EachConnectedLineIndividually[i]], ['Lines with connected components and no horizontal spaces'])
        print(EachConnectedLineIndividually[i].shape)

    # Super Texture Block Generation
    superTextureBlock = np.ones((maxHeightsSum, np.max(WidthsforLines)))
    print(superTextureBlock.shape)
    yprevious = 150
    Begining2 = 0
    for i in range(countNew):
        Begining2 = abs(int((EachConnectedLineIndividually[i].shape[0]) / 2) - int(yprevious / 2))

        superTextureBlock[Begining2:EachConnectedLineIndividually[i].shape[0] + Begining2,
        0:EachConnectedLineIndividually[i].shape[1]][EachConnectedLineIndividually[i] == 0] = \
            superTextureBlock[Begining2:EachConnectedLineIndividually[i].shape[0] + Begining2,
            0:EachConnectedLineIndividually[i].shape[1]][EachConnectedLineIndividually[i] == 0]

        superTextureBlock[Begining2:EachConnectedLineIndividually[i].shape[0] + Begining2,
        0:EachConnectedLineIndividually[i].shape[1]][EachConnectedLineIndividually[i] != 0] = \
            EachConnectedLineIndividually[i][EachConnectedLineIndividually[i] != 0]

        yprevious = yprevious + avgHeights[i] / 2

    # show_images([superTextureBlock[0:int(EachConnectedLineIndividually[-1].shape[0] + Begining2), :]],
    #             ['Lines with connected components and no horizontal spaces'])
    return superTextureBlock[0:int(EachConnectedLineIndividually[-1].shape[0] + Begining2), :]
