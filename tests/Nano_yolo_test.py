import os
import shutil

import sys
from pathlib import Path
import random as rd
import glob
import cv2
ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in range(100):
        if (len(glob.glob(str(new_path / ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
#
root = get_root()
#

path_to_yolo = Path(glob.glob(str(root)+"/**/*Nano_yolo.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_yolo))
import Nano_yolo as NY
#
def basic_run_test():
    # runs ml model and checks for any classification
    path_to_picture =  Path(glob.glob(str(root)+"/**/*images/train/18.jpg",recursive = True)[0]).parent.absolute()
    possible_files = os.listdir(str(path_to_picture))
    pick_me = rd.randint(0,len(possible_files)-1)
    print("fname = ",str(path_to_picture/str(possible_files[pick_me])),"\n\n\n")
    img = cv2.imread(str(path_to_picture/str(possible_files[pick_me])))
    obj_det = NY.CustomYolo(config_path=str(root / "src" / "config_distance.yml"), DEBUG = True, saveDir = "runs")
    ret_val = obj_det.formate_image_get_mm(img)
    success = len(ret_val)>0
    return success
def merge_test():
    path_to_picture =  Path(glob.glob(str(root)+"/**/*img_test_merge.png",recursive = True)[0])
    img = cv2.imread(str(path_to_picture))
    cx = 100
    cy = 100
    w = 100
    h = 100
    cx1 = 200
    cy1 = 200
    w1 = 100
    h1 = 100
    returnTup = NY.merge_if_overlapping(cx, cy, w, h,
                            cx1, cy1, w1, h1,
                            x_padding=0.05,DEBUG = True,img=img)
    success = returnTup[0]
    return success
def merge_all():
    path_to_picture =  Path(glob.glob(str(root)+"/**/*img_test_merge.png",recursive = True)[0])
    img = cv2.imread(str(path_to_picture))
    obj_det = NY.CustomYolo(config_path=str(root / "src" / "config_distance.yml"), DEBUG = False, saveDir = "runs")
    ret_val = obj_det.formate_image_get_mm(img)
    print(ret_val)
    new_list  = NY.merge_too_close(ret_val,DEBUG = True,img = img)

    return ( len(ret_val)>=len(new_list)>0)
if __name__ =="__main__":
    print("basic run test ")
    result = basic_run_test()
    result = merge_test()
    result = merge_all()
    print("result:", result)
