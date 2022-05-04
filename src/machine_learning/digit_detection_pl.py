import glob
import sys
import cv2
# custom
import image_methods as i_m
import trained_model as dr
import ExtractDigits as ex
import FindChipsAndNums as chan

ROOT = ".git"

isWindows = sys.platform.startswith('win')
pathChar = "/"
if isWindows:
    pathChar = "\\"


digit_weights = glob.glob("*.pth")[0]


def extract_dr(img):
    digits, found = ex.find_digits(img)
    if found:
        out = ""
        for digit in digits:
            pillow_im = i_m.cv2_to_PIL(digit)
            prediction = dr.classify(digit_weights, pillow_im)

            out = str(int(prediction)) + out
        return out
    return "-1"


def chip_pred(img):
    num_strip = chan.crop_num_rec(img)
    return extract_dr(num_strip)


def trav_img(img):
    chip_num = ""
    extracted_num = chan.find_single_num(img)
    digits, _ = ex.find_digits(extracted_num)
    for digit in digits:
        digit = i_m.cv2_to_PIL(digit)
        num = dr.predict_num(digit)
        chip_num += str(int(num))
    return int(chip_num)


if __name__ == "__main__":
    images = glob.glob("chips_in_traveler/*.png")
    for image in images:
        im = cv2.imread(image)
        print("Predicted: ", trav_img(im))
# "./chips_in_traveler/2000.png"
