import cv2
import numpy as np
import math
import sys
import os
isWindows = sys.platform.startswith('win')
pathChar = "/"
if(isWindows): pathChar = "\\"

def parseChipImage(dirPath=".{p}dev{p}generateTestImagesMacro{p}Assets{p}".format(p=pathChar)):
    listOfImages = []
    for i in range(4):
        path = "/home/whorehay/Desktop/github/NanoView_G33/"+dirPath+"C{i}.png".format(i=i+1)
        print(path)
        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        listOfImages.append(gray)
    return listOfImages
def getSquares(listOfchips):
    listOfnums = []
    widthPercentage = 0.45
    heightPercentage = 0.50
    squareStartx = 0.27
    squareStarty = 0.75
    listOfNumsWithChips = []
    count = 0
    for i in listOfchips:
        test = i
        [h,w] = test.shape
        sw = math.ceil(w*squareStartx)
        sh = math.ceil(h*squareStarty)
        cw = sw+math.ceil(w*widthPercentage)
        ch = sh+math.ceil(h*heightPercentage)
        t1 = test[sh:,sw:cw]
        widthNum = 0.33
        listOfVectorized = []
        for j in range(3):
            [h,w] = t1.shape
            d1 = math.floor(w*widthNum*j)
            d2 = math.floor(w*widthNum*(j+1))
            n1 = t1[:,d1:d2]
            n1 = cv2.resize(n1,[128,128], interpolation = cv2.INTER_AREA)
            n2 = cv2.threshold(n1,68, 255, cv2.THRESH_BINARY)[1]
            blurred = cv2.GaussianBlur(n2, (3, 3), 0)
            blurred = cv2.bitwise_not(n2)
            inVecFor = np.reshape(n1 ,(128*128,1))
            listOfVectorized.append(inVecFor)
            count+=1
            # cv2.imshow("s",n1  )
            # cv2.waitKey(0)
            cv2.imwrite("/home/whorehay/Desktop/github/NanoView_G33/dev/generateTestImagesMacro/Assets/"+str(count)+".png",n1)
        arr = np.array(listOfVectorized)
        listOfNumsWithChips.append(arr)
    chipsByNumsByIm = np.array(listOfNumsWithChips)

    return chipsByNumsByIm
def getMeans(dirPath=".{p}dev{p}generateTestImagesMacro{p}Assets{p}generated_numbers{p}".format(p=pathChar)):
    listOfImages = []
    for i in range(6):
        img = cv2.imread(dirPath+f"Base_{i}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row,col= img.shape
        # # change colors

        # img = img1*0.8 + 10
        # img = img.astype(np.uint8)
        # img = cv2.GaussianBlur(img, (19, 19), 0)
        # img1 = img.astype(np.float64)
        # # cv2.imshow("k",img)
        # # cv2.waitKey(0)
        #  #add noise
        # mean = 2
        # var = 0.1
        # sigma = var**0.5
        # gauss = np.random.normal(mean,sigma,(row,col))
        # gauss = gauss.reshape(row,col)
        # nois = gauss.astype(np.float64)*img1
        # # nois = nois+gauss
        # img = cv2.GaussianBlur(nois.astype(np.uint8), (19, 19), 0)
        cv2.imshow("k",img)
        cv2.waitKey(0)
        # #blur

        # cv2.imshow("k",img)
        # cv2.waitKey(0)
        inVecFor = np.reshape(img,(128*128,1))
        listOfImages.append(inVecFor)
    means = np.array(listOfImages)
    return means
def getNearest(vec, means):
    n = means.shape[0]
    minD = 100000
    minInd = -1;
    # print(means.shape)
    # print(vec.shape)
    minvec = vec
    for i in range (n):
        s1 = means[i,:,:]
        d = np.linalg.norm(s1-vec)
        if(d<minD):
            minD = d
            minvec = s1
            # print("found one")
            minInd = i
    cv2.imshow("guess",np.reshape(minvec ,(128,128,1)))
    cv2.waitKey(0)
     
    cv2.imshow("real",np.reshape(vec ,(128,128,1)))
    cv2.waitKey(0)

    return minInd

def kNearest(vecs,means):
    numImages =vecs.shape[0]
    numNumbersPerImage = vecs.shape[1]
    for i in range(numImages):
        numList = []
        for j in range(numNumbersPerImage):
            listOfNums = vecs[i,j,:,:]
            ind = getNearest(listOfNums,means)
            numList.append(ind)
        print(str(numList ))

# k = parseChipImage()

def readLabels():
    path = "/home/whorehay/Desktop/github/NanoView_G33/dev/generateTestImagesMacro/Assets/meansForNN/"
    files = os.listdir(path)


    listicules = [[],[],[],[],[]]
    for f in files:
            img = cv2.imread(path+f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow(f,img)
            # cv2.waitKey(0)
            label = f[-5:-4]
            if(label =="0"):
                listicules[0].append(img)
            elif(label =="1"):
                listicules[1].append(img)
            elif(label =="2"):
                listicules[2].append(img)
            elif(label =="3"):
                listicules[3].append(img)
            elif(label =="5"):
                listicules[4].append(img)
    means=[]
    for l in listicules:
        classes = np.array(l)
        mean = np.mean(classes,axis=0).astype('uint8')
        cv2.imshow('cats',mean)
        cv2.waitKey(0)
        # print(mean.shape)
        mean = np.reshape(mean ,(128*128,1))
        means.append(mean)
    k = np.array(means)
    return k
    #
    # print(zeros.shape)
    # print(zeros[0].shape)
    # avZero = np.mean(zeros,axis=0)
    # print(np.mean(avZero))
    # cv2.imshow("hahs",avZero)
    # cv2.waitKey(0)
    # for img in LP_0:
    #     cv2.imshow("haha",img)
    #     cv2.waitKey(0)
k = parseChipImage()
meanses = readLabels()
vecs = getSquares(k)
print(meanses.shape  ,vecs.shape)
# means = getMeans()

kNearest(vecs,meanses)
