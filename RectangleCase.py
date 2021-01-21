from Imports import *
from detectLines import detect_lines


def GetTextureBlock(img):
    grey = rgb2gray(img)
    thr = threshold_otsu(grey)
    croppedBinarized = np.copy(grey)
    croppedBinarized[grey < thr] = 0
    croppedBinarized[grey >= thr] = 1
    temporary = []
    temporary = detect_lines(croppedBinarized)
    Upper = temporary[-2]
    Lower = temporary[-1]
    cropped = grey[Upper + 5:Lower - 5, 50:]
    croppedBinarized = croppedBinarized[Upper + 5:Lower - 5, 50:]
    # ------------------------------------------------

    croppedBinarized = 1 - croppedBinarized
    kernel = np.ones((1, 150), np.uint8)
    dilatedImage = cv2.dilate(croppedBinarized, kernel, iterations=1)
    closingImage = cv2.erode(dilatedImage, kernel, iterations=1)
    croppedBinarized = 1 - croppedBinarized
    closingImage = 1 - closingImage

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
            # print(area)
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

        if (tempcoord.shape[0] * tempcoord.shape[1] > 40000):
            EachLineIndividually.append(tempcoordMultipliedByBinarized)
            EachLineIndividuallyBinarized.append(tempcoordBinarized)
            count += 1
    EachLineIndividually = np.asarray(EachLineIndividually)

    countNew = 0
    EachConnectedLineIndividually = []
    avgHeights = []
    maxHeightsSum = 0
    WidthsforLines = []
    maxHeightsforLines = []
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
            if area > 100:
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
                # print("aaaaaaaaaa")
                ConnectedBlocks = ConnectedBlocks[0:maxHeightofLine, 0:Width]
                EachConnectedLineIndividually.append(ConnectedBlocks)
                countNew += 1
                ConnectedBlocks = np.zeros(EachLineIndividually[i].shape)
                previousX = 0
                maxHeightsforLines.append(maxHeightofLine)
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
        if ConnectedBlocks.shape[0] * ConnectedBlocks.shape[1] > 10000:
            EachConnectedLineIndividually.append(ConnectedBlocks)
            countNew += 1
            maxHeightsforLines.append(maxHeightofLine)
            maxHeightsSum += maxHeightofLine
            avgHeights.append(avgHeight / int(countForAvgheight + 1))
            WidthsforLines.append(sumofWidthsofEachLine)

    superTextureBlock = np.ones((maxHeightsSum, np.max(WidthsforLines)))
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

    superTextureBlock = superTextureBlock[0:int(EachConnectedLineIndividually[-1].shape[0] + Begining2), :]

    start = int(150 / 2) - int(maxHeightsforLines[0] / 2)
    if start < 0:
        start = 0
    end = int(yprevious) - int(avgHeights[-1] / 2) + start + int(maxHeightsforLines[-1] / 2)

    superTextureBlock = superTextureBlock[0:end, :]
    supertextureBlockCopy = np.copy(superTextureBlock)
    textureCounter = 0
    textureBlocks = []

    aspectRatio = (supertextureBlockCopy.shape[1]) / (supertextureBlockCopy.shape[0])

    supertextureBlockCopy1 = resize(supertextureBlockCopy, (int((9 * 256) / (aspectRatio)), 9 * 256),
                                    anti_aliasing=False)

    if supertextureBlockCopy1.shape[0] < 256:
        supertextureBlockCopy1 = resize(supertextureBlockCopy, (256, int((256) * (aspectRatio))), anti_aliasing=False)

    if supertextureBlockCopy1.shape[1] < 9 * 256:
        supertextureBlockCopy1 = resize(supertextureBlockCopy, (256, 9 * 256), anti_aliasing=False)

    supertextureBlockCopy = supertextureBlockCopy1

    textureHeight = int(np.floor(supertextureBlockCopy.shape[0] / 2))
    for i in range(4):
        textureBlocks.append(
            supertextureBlockCopy[textureHeight - 128:textureHeight, textureCounter:textureCounter + 256])
        textureBlocks.append(
            supertextureBlockCopy[textureHeight:textureHeight + 128, textureCounter:textureCounter + 256])
        textureCounter += 256

    textureBlocks.append(supertextureBlockCopy[textureHeight - 128:textureHeight, textureCounter:textureCounter + 256])

    return textureBlocks
