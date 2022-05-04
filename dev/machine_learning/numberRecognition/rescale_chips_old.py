import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os
from pathlib import Path
import sys
## used for training ML to turn wide chip images to scannable
ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split("/"):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
root = get_root()

path_to_im_methods = Path(glob.glob(str(root)+"/**/*image_methods.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_im_methods))
import image_methods as wrp

def deNoise(grey_img):
    kernelSizes = [(3, 3), (9, 9), (15, 15)]
    blurred = grey_img
    for (kX, kY) in kernelSizes:
        blurred = cv2.blur(blurred, (kX, kY))
    return blurred


def pre_process(img):  # input color image, output grayscale img

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # set a thresh
    thresh = 150
    # get threshold image
    deNoised = deNoise(img_grey)
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    cont, out1 = wrp.increaseContrast(cv2.cvtColor(thresh_img,cv2.COLOR_GRAY2RGB))
    kernel = np.ones((15,15), np.uint8)

    img_erosion = cv2.erode(cv2.cvtColor(thresh_img,cv2.COLOR_GRAY2RGB), kernel, iterations=1)
    img_dilation = cv2.dilate(cv2.cvtColor(thresh_img,cv2.COLOR_GRAY2RGB), kernel, iterations=1)
    img_dilation_grey = cv2.cvtColor(img_dilation,cv2.COLOR_BGR2GRAY)

    # cv2.imshow("hah",cont)
    # cv2.imshow("hah1",img_erosion)
    # cv2.imshow("ha1",img_dilation)
    # cv2.imshow("haha1",img_dilation_grey)
    # cv2.waitKey(0)
    return img_dilation_grey
def warpImage(img,pts = np.float32([[ 821 , 482 ],[ 1574 , 578 ],[ 691 , 1950 ],[ 1907 , 1917 ],]) ,x = 512,y=512):
    pts2 = np.float32([[0,0],[x,0],[0,y],[x,y]])
    mat = cv2.getPerspectiveTransform(pts,pts2)
    # print(mat)
    out = cv2.warpPerspective(img, mat, (x, y))
    return out


def findContours(img):
    img_processed = pre_process(img)

    # find contours
    imagem = cv2.bitwise_not(img_processed)
    cnts = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # create an empty image for contours
    img_contours = img.copy()  # np.zeros(img.shape)
    # draw the contours on the empty image

    totalArea = img.shape[0] * img.shape[1]
    # print(len(cnts),totalArea )
    newImages = []
    for c in cnts[0]:  # compute the center of the contour
        M = cv2.moments(c)  # find the center of a given contour
        Area = cv2.contourArea(c)  # find the area of the contour

        if  totalArea*.05<Area < totalArea*.30:  # this is bad.!!!!!!!!!!!!!!!!!!!!!! This needs to be tuned
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            rect = cv2.minAreaRect(c)
            x, y, w, h = cv2.boundingRect(c)
            rect = [x, y, x + w, y + h]
            cropped = img[rect[1]:rect[3], rect[0]:rect[2]]
            boxDim = np.sqrt(totalArea * .07)
            rect1 = ((cX, cY), (boxDim, boxDim), rect[2])
            box = cv2.boxPoints(rect1)  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            # print(box,img.shape)
            cv2.rectangle(img_contours, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.drawContours(img_contours, [box], 0, (0, 0, 255), 10)
            cv2.circle(img_contours, (cX, cY), 20, (255, 0, 255), -1)

            newImages.append(cropped)
    # cv2.imshow("hah",img_processed)
    # cv2.imshow("hah1",img)
    # cv2.imshow("ha1",img_contours)

    cv2.waitKey(0)
    return img_contours,  newImages


def save_chips(chips):
    for i, chip in enumerate(chips):
        cv2.imwrite(str(root/"dev"/ "machine_learning"/ "numberRecognition"/ "chips_output"/ f"chip{i}.jpg"), chip)


def iterate_over_folder():
    p_2Img = str(root/"chip_photos")
    all_chips = []
    chip_pic_paths = os.listdir(p_2Img)
    for pics in chip_pic_paths:
        img = cv2.imread(str(root/"chip_photos"/pics))
        cont, outputs, chips = findContours(img)
        all_chips = all_chips + chips

    for chip in all_chips:
        cv2.imshow("chip",chip)
        cv2.waitKey(1000)
    # save_chips(all_chips)
if __name__ =="__main__":
    iterate_over_folder()

    # img = cv2.imread("./chip_img_movement/output_imgs/csv33.jpg")
    # cont,outputs,chips = findContours(img)
    # for i in chips:
    #     outputs+=[["chip",i]]
    # wrp.showImageList(outputs)
