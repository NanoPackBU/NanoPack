
import glob
import sys
from pathlib import Path
import cv2
from FindChipsAndNums import find_single_num
from ExtractDigits import find_digits
from ExtractDigits import extract_digits
from trained_model import predict_num
from PIL import Image
from matplotlib import cm
import numpy as np


ROOT = ".git"

isWindows = sys.platform.startswith('win')
pathChar = "/"
if (isWindows): pathChar = "\\"

def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split(pathChar):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path

root = get_root()
## used for getting the whole number recog pipeline
path_to_im_methods = Path(glob.glob(str(root)+"/**/*image_methods.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_im_methods))
import image_methods as wrp

path_to_trained_model = Path(glob.glob(str(root)+"/**/*trained_model.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_trained_model))
import trained_model as dr

path_to_digit_extract = Path(glob.glob(str(root)+"/**/*ExtractDigits.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_digit_extract))
import ExtractDigits as ex

path_to_digit_find_nums = Path(glob.glob(str(root)+"/**/*FindChipsAndNums.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_digit_find_nums))
import FindChipsAndNums as chan

digit_weights = glob.glob(str(root)+"/**/*.pth",recursive = True)[0]

def extract_dr(img):
    digits, found = ex.find_digits(img)
    if found:
        out = ""
        for im in digits:
            pillow_im = wrp.cv2_to_PIL(im)
            prediction = dr.classy(digit_weights ,pillow_im)

            out = str(int(prediction))+out
        return out
    return "-1"

def chip_pred(img):
    num_strip = chan.crop_num_rec(img)
    return extract_dr(num_strip)

def trav_img(img):
    chip_num = ""
    extracted_num = chan.find_single_num(img)
    digits, _, _ = find_digits(extracted_num)
    for digit in digits:
        digit = wrp.cv2_to_PIL(digit)
        num = predict_num(digit)
        chip_num += str(int(num))
    return int(chip_num)
    

if __name__ == "__main__":
    images = glob.glob("chips_in_traveler/*.png")
    for image in images:
        img = cv2.imread(image)
        print(trav_img(img))
# "./chips_in_traveler/2000.png"

