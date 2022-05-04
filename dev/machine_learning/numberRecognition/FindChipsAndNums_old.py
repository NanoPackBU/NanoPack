import numpy as np
import glob
import cv2
import os
import sys
from pathlib import Path

isWindows = sys.platform.startswith('win')
pathChar = "/"
if isWindows:
    pathChar = "\\"
startingdir = os.getcwd() + pathChar

ROOT = ".git"


def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split(pathChar):
        if len(glob.glob(str(new_path / ROOT))) > 0:
            break
        new_path = new_path.parent.absolute()
    return new_path


#
root = get_root()
#
path_to_yolo = Path(glob.glob(str(root) + "/**/*image_methods.py", recursive=True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_yolo))
import image_methods as im


def overlap_checker(rect1, rect2):
    pad = 10
    if (rect1[0] - pad >= rect2[2] + pad) or (rect1[2] + pad <= rect2[0] - pad) or (
            rect1[3] + pad <= rect2[1] - pad) or (rect1[1] - pad >= rect2[3] + pad):
        return False
    else:
        return True


def contains_checker(rect1, rect2):
    if (rect1[0] > rect2[0]) and (rect1[1] > rect2[1]) and (rect1[2] < rect2[2]) and (rect1[3] < rect2[3]):
        return True
    elif (rect1[0] < rect2[0]) and (rect1[1] < rect2[1]) and (rect1[2] > rect2[2]) and (rect1[3] > rect2[3]):
        return True
    return False


def merge(rect1, rect2):
    new_rect = [0, 0, 0, 0]
    if rect1[0] < rect2[0]:  # keep smaller
        new_rect[0] = rect1[0]
    else:
        new_rect[0] = rect2[0]
    if rect1[1] < rect2[1]:  # keep smaller
        new_rect[1] = rect1[1]
    else:
        new_rect[1] = rect2[1]
    if rect1[2] > rect2[2]:  # keep larger
        new_rect[2] = rect1[2]
    else:
        new_rect[2] = rect2[2]
    if rect1[3] > rect2[3]:  # keep larger
        new_rect[3] = rect1[3]
    else:
        new_rect[3] = rect2[3]
    return new_rect


def find_chips(img):
    img = img[0:200, 75:275]
    cv2.imshow('img', img)
    cv2.waitKey(5000)
    img_copy = img

    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = im.green_to_white(img, [150, 255, 150], [0, 120, 0])

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)

    img = final

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(gray, kernel, iterations=10)
    img = cv2.erode(img, kernel, iterations=10)

    image2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    # image = bruteForceFindSquares(image2,image1)

    img = cv2.dilate(image2, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = ~img
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(img, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.merge([img, img, img])
    rects = []

    offset_x = 0
    offset_y = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = [x, y, x + w, y + h]
            if x + w < 215:
                rects.append(rect)
            img = cv2.rectangle(img, (x - offset_x, y - offset_y), (x + w + offset_x, y + h + offset_y), (255, 0, 0), 3)
            # print("here", x - offset_x, y - offset_y)
            # print(x + w + offset_x, y + h + offset_y)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    converged = False
    # print(len(rects))
    while not converged:
        converged = True
        length = len(rects)
        for i in range(length - 1):
            for j in range(i + 1, length - 0):
                if overlap_checker(rects[i], rects[j]):
                    area1 = (rects[i][3] - rects[i][1]) * (rects[i][2] - rects[i][0])
                    area2 = (rects[j][3] - rects[j][1]) * (rects[j][2] - rects[j][0])
                    if area1 < 30000 or area2 < 30000:
                        rects[i] = merge(rects[i], rects[j])
                        rects.remove(rects[j])
                        converged = False
                        break
            if not converged:
                break
    # print(len(rects))

    converged = False
    while not converged:
        converged = True
        length = len(rects)
        for i in range(length - 1):
            for j in range(i + 1, length):
                if contains_checker(rects[i], rects[j]):
                    rects[i] = merge(rects[i], rects[j])
                    rects.remove(rects[j])
                    converged = False
                    break
            if not converged:
                break

    final_rects = []
    for rect in rects:
        # print(rect[0], rect[1], rect[2], rect[3], (rect[2] - rect[0]) * (rect[3] - rect[1]))
        if (rect[2] - rect[0]) * (rect[3] - rect[1]) > 25000:
            final_rects.append(rect)
            img_copy = cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]),
                                     (255, 0, 0), 3)
    return img_copy, final_rects


def crop_to_chip(img, rect):
    # print(img.shape)
    # print(rect)
    l = rect[1]
    r = rect[3]
    t = rect[0]
    b = rect[2]
    # print(l,r,t,b)
    cropped = img[rect[1]:rect[3], rect[0]:rect[2]]
    return cropped


def crop_num_rec(chip_img):
    chip_height, chip_width, _ = chip_img.shape
    num = chip_img[int(chip_height * 0.76):int(chip_height * 0.97),
          int((chip_width / 4) * 0.9):int((chip_width * 3 / 4))]
    return num


def crop_chips(chips_img, rects, count=0, out_folder="", write=True):
    img = chips_img
    num = None 
    cropped = None
    for rect in rects:
        cropped = crop_to_chip(img, rect)
        num = crop_num_rec(cropped)
        if (write):
            name = startingdir + "chip_outputs" + pathChar + "chip_" + str(count) + ".png"
            num_name = startingdir + out_folder + pathChar + "num_" + str(count) + ".png"
            cv2.imwrite(name, cropped)
            cv2.imwrite(num_name, num)
            count += 1
        # cv2.imshow("num", num)
        # cv2.waitKey(0)
    return num, cropped, len(rects)


def find_nums(in_folder, out_folder):
    chip_count = 1
    pic_locs = glob.glob(in_folder)
    for loc in pic_locs:
        chips_photo = startingdir + loc
        img = cv2.imread(chips_photo)
        image, rectangles = find_chips(img)
        chips_img = cv2.imread(chips_photo)
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        num_rects = crop_chips(image, rectangles, chip_count, out_folder, write=True)
        chip_count += num_rects


def find_single_chip_nums(pic, out_folder, count):
    image, rectangles = find_chips(pic)
    num_rects = crop_chips(pic, rectangles, count, out_folder)


def find_single_num(pic):
    image, rects = find_chips(pic)
    num, _, num_rects = crop_chips(image, rects, write=False)
    return num


if __name__ == "__main__":
    find_nums('chips_in_traveler/*.png', "num_outputs")
