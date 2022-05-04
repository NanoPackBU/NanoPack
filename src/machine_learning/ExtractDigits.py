import cv2
import numpy as np
import sys
import os
import glob

isWindows = sys.platform.startswith('win')
pathChar = "/"
if isWindows:
    pathChar = "\\"
startingdir = os.getcwd() + pathChar


# Input the image with the full chip number
def find_digits(img):
    out_img = img.copy()
    img = ~img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # im, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_OTSU)

    # _, thresh2 = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    thresh2 = np.pad(thresh2, [(50, 25), (50, 25)], mode="constant", constant_values=255)

    #################      Now finding Contours         ###################

    thresh2 = ~thresh2
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    for cnt in contours2:
        # print("area = ", cv2.contourArea(cnt))
        if 500 > cv2.contourArea(cnt) > 75:
            [x, y, w, h] = cv2.boundingRect(cnt)
            # removing padding

            x = x - 50
            y = y - 50
            in_h, in_w = img.shape[:2]
            w = w if (x + w) < in_w else in_w - x
            h = h if (y + h) < in_h else in_h - y
            digit = [x, y, w, h]
            # print("h = ", h, "x + w = ", x + w, "x = ", x)

            if 20 <= h <= 30 and (x > 0 or x + w >= 20):
                # cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # cv2.imshow('thresh', out_img)
                # cv2.waitKey(0)
                # print(digit)
                digits.append(digit)
    digits.sort()

    if len(digits) != 3:
        cv2.imshow('original', img)
        cv2.imshow('contours drawn', out_img)
        cv2.imshow('thresh', thresh2)
        cv2.waitKey(1000)

    img_digits = []
    for i in digits:
        img_done = crop_dimensions(out_img, i)
        img_digits.append(img_done)

    gottem = len(digits) == 3
    return img_digits, gottem


def crop_dimensions(img, dims):
    l = dims[0]
    r = dims[0] + dims[2]
    t = dims[1]
    b = dims[1] + dims[3]
    return img[t:b, l:r, :]


def extract_digits(in_folder, out_folder, write=True):
    digit_count = 1
    num_locs = glob.glob(in_folder)
    missed_it = 0
    for loc in num_locs:
        num_photo = startingdir + pathChar + loc
        img = cv2.imread(num_photo)
        digits, gottem = find_digits(img)
        if not gottem:
            missed_it += 1
        if len(digits) <= 0:
            continue
        if write:
            for digit in digits:
                name = startingdir + out_folder + pathChar + "digit_" + str(digit_count) + ".png"
                cv2.imwrite(name, digit)
                digit_count += 1
    print("you fucked up : ", missed_it, f"that's {int((missed_it / len(num_locs)) * 100)}% for you buddy")


if __name__ == "__main__":
    print("find digits ")
    extract_digits("num_outputs/*.png", "digit_outputs")
