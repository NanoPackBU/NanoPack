# this file generates random arangements of a traveler for the image rec
import sys

sys.path.append("/home/pequode/.local/lib/python3.8/site-packages")  # support for gks editor which doesnt like external models
import cv2
import math
import random as rd

isWindows = sys.platform.startswith('win')
pathChar = "/"
if isWindows: pathChar = "\\"
NanoviewPath = "."
# NanoviewPath = "YOUR PATH HERE"
# bruteforce chipid global Remove for prod


# this function puts an image with an alpha channel ontop of another image

# this creates a list of random numbers that are non repeating... I now realize that there is a random function that does this
def GenListOfInds(chipNumber):
    listChipPlaces = []
    k = chipNumber
    while k > 0:
        x = rd.randint(0, 7)
        y = rd.randint(0, 7)
        tup = (x, y)
        if tup not in listChipPlaces:
            listChipPlaces.append(tup)
            k -= 1
    return listChipPlaces


# this reads in chip images to be placed and returns a list of chip images at a consistent size
def createchiplist():
    ChipList = []
    for i in range(4):
        path_p1 = pathMake() + "Assets{p}C".format(p=pathChar)

        C1 = cv2.imread(path_p1 + "{num}.png".format(num=i + 1), cv2.IMREAD_UNCHANGED)
        C1 = cv2.resize(C1, [155, 155], interpolation=cv2.INTER_AREA)
        ChipList.append(C1)
    return ChipList


def populateTraveler(travelers, chip, randPos=False, maxChips=8, minchips=3):
    # this function is big
    scalFac = 11  # scales up the traveler
    iHO = 0.11  # the displacement between actual chips y
    iWO = 0.11  # the displacement between actual chips x
    IOFY = 0.13  # where the chip slots start on the traveler in y
    IOFX = 0.1  # where the chip slots start on the traveler in x

    (h, w) = travelers.shape[:2]
    traveler = cv2.resize(travelers, [math.floor(h * scalFac), math.floor(w * scalFac)], interpolation=cv2.INTER_AREA)
    # scales up traveler by the amount

    (th, tw) = traveler.shape[:2]
    initHOffset = math.floor(th * iHO)
    initWOffset = math.floor(tw * iWO)
    eachOffsetx = math.floor(tw * IOFX)
    eachOffsety = math.floor(tw * IOFY)
    PlacedTraveler = traveler
    chipNumber = rd.randint(minchips, maxChips)
    chipLocsCsvfor = "ChipNumber,ChipID,ClamshellGroup\n"
    if (randPos):
        inds = GenListOfInds(chipNumber)
        for n, i in enumerate(inds):
            m = n
            x = i[0]
            y = i[1]
            chipid = m % len(chip) - 1
            chips = chip[chipid]
            chipLocsCsvfor += str(8 * x + y) + "," + str(chipNam[chipid]) + "," + str(rd.randint(1, 16)) + "\n"
            (PlacedTraveler, a) = overlay_image_alpha(PlacedTraveler, chips, initWOffset + x * eachOffsetx,
                                                      initHOffset + y * eachOffsety)
    else:
        for i in range(8):
            for j in range(8):
                chips = chip[i * j + j % len(chip) - 1]
                (PlacedTraveler, a) = overlay_image_alpha(PlacedTraveler, chips, initWOffset + i * eachOffsetx,
                                                          initHOffset + j * eachOffsety)
                chipNumber -= 1;
                if (chipNumber <= 0): break
            if (chipNumber <= 0): break
    return PlacedTraveler, chipLocsCsvfor


def randomRotateTraveler(minchips=3, maxChips=8):
    angle = rd.randint(-10, 10)
    traveler1 = cv2.imread(pathMake() + "Assets{p}traveler.png".format(p=pathChar), cv2.IMREAD_UNCHANGED)
    traveler2, csvtext = populateTraveler(traveler1, createchiplist(), randPos=True, maxChips=maxChips,
                                          minchips=minchips)
    traveler3 = RotateImageSafe(traveler2, angle)
    return traveler3, csvtext


def randomRotateClamshell():
    angle = rd.randint(-10, 10)
    clamShell = cv2.imread(pathMake() + "Assets{p}ClamShell.png".format(p=pathChar), cv2.IMREAD_UNCHANGED)
    rC = RotateImageSafe(clamShell, angle)
    return rC


def randoPlaceTraveler(background, minchips=3, maxChips=8):
    scaleFactor = 2
    padding = 0.015
    t3ScaleFac = 0.33
    traveler3, csvtext = randomRotateTraveler(minchips=minchips, maxChips=maxChips)
    (tw, th) = background.shape[:2]
    traveler3 = cv2.resize(traveler3, [math.floor(tw * t3ScaleFac), math.floor(t3ScaleFac * th)],
                           interpolation=cv2.INTER_AREA)
    (tw, th) = traveler3.shape[:2]
    biggerDir = tw if tw > th else th
    canvasWidth = math.floor(biggerDir * scaleFactor)

    xoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - tw))
    yoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - th))
    traveler4, _ = overlay_image_alpha(background, traveler3, xoff, yoff)

    return csvtext, traveler4, (xoff, yoff, tw, th)


def randoPlaceClam(background, dimsTrav):
    clamShell = randomRotateClamshell()
    scaleFactor = 2
    padding = 0.015
    ScaleFacFromBack = 0.15
    (tw, th) = background.shape[:2]
    scaledClam = cv2.resize(clamShell, [math.floor(tw * ScaleFacFromBack), math.floor(ScaleFacFromBack * th)],
                            interpolation=cv2.INTER_AREA)

    (tw, th) = scaledClam.shape[:2]
    biggerDir = tw if tw > th else th
    canvasWidth = math.floor(biggerDir * scaleFactor)
    xoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - tw))
    yoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - th))
    while (Intersecting(xoff, yoff, tw, th, dimsTrav[0], dimsTrav[1], dimsTrav[2], dimsTrav[3])):
        xoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - tw))
        yoff = rd.randint(math.floor(canvasWidth * padding), math.floor(canvasWidth - canvasWidth * padding - th))

    randoPlaceClam, _ = overlay_image_alpha(background, scaledClam, xoff, yoff)
    return randoPlaceClam, (xoff, yoff, tw, th)


def writeImg(outPath, minchips=8, maxChips=8):
    background = makeCanvas(1000, 1000, pathMake() + "Assets{p}base.jpeg".format(p=pathChar))
    randomized = randomPass(background)
    [csv, img_out, loc] = randoPlaceTraveler(randomized, minchips=minchips, maxChips=maxChips)
    clam, clamloc = randoPlaceClam(img_out, loc)

    cv2.imwrite(outPath, clam)
    return csv


def generateTests(number, persentTrain=0.8, presentTest=0.2, minchips=3, maxChips=8):
    sampTrain = math.floor(number * persentTrain)
    sampTest = number - sampTrain
    for i in range(sampTrain):
        path = pathMake() + "Output_Images{p}Train_{a}.png".format(p=pathChar, a=i + 1)
        csv = writeImg(path, minchips=minchips, maxChips=maxChips)
        pathcsv = pathMake() + "SampleCSV{p}Train_{a}.csv".format(p=pathChar, a=i + 1)
        f = open(pathcsv, "w")
        f.write(csv)
        f.close()

    for i in range(sampTest):
        path = pathMake() + "Images{p}Test_{a}.png".format(p=pathChar, a=i + 1)
        csv = writeImg(path, minchips=minchips, maxChips=maxChips)
        pathcsv = pathMake() + "SampleCSV{p}Test_{a}.csv".format(p=pathChar, a=i + 1)
        f = open(pathcsv, "w")
        f.write(csv)
        f.close()


# generateTests(1,minchips = 30,maxChips=62)


if __name__ == "__main__":
    generateTests(15)
    # imgp = "/home/pequode/Pictures/Sentement/IMG_0412.JPG"
    # im = cv2.imread(imgp)
    # im = cv2.resize(im,[700, 700], interpolation = cv2.INTER_AREA)
    # cv2.imshow("prefilter",im)
    # cv2.waitKey(0)
    # im = randomGraidentlight(im,10)
    # cv2.imshow("postfilter",im)
    # cv2.waitKey(0)
