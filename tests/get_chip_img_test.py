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
        if(new_path == new_path.parent.absolute()): return "not fond"
        new_path = new_path.parent.absolute()
    return new_path
#
root = get_root()

#
path_to_get_chip= Path(glob.glob(str(root)+"/**/*get_chip_img.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_get_chip))
import get_chip_img as gc

def basic_run_test():
    mover = gc.chip_imgr(str(Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0])))
    print("working")
    return True
def readImg_test():
    import camera_spoofer as cap
    mover = gc.chip_imgr(str(Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0])))
    cam_wide = cap.Fake_cam(0)
    img = mover.readImg(cam_wide,2)
    if img.shape[0] >0 and img.shape[1] >0:
        return True
    return False
def find_general_loc_test():
    import camera_spoofer as cap
    mover = gc.chip_imgr(str(Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0])))
    path_to_ml_test_img = str(Path(glob.glob(str(root)+"/**/*images/train/1.jpg",recursive = True)[0]))
    cam_wide = cap.Fake_cam(0,cv2.imread(path_to_ml_test_img))
    output  = mover.find_general_loc(cam_wide)
    return len(output)>0
def find_top_left_test():
    import camera_spoofer as cap
    import tinyg_spoofer as tinyg
    center = [0,0]
    mover = gc.chip_imgr(str(Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0])))
    tgo = tinyg.tinyg_obj()
    cam_short = cap.Fake_cam(0)
    try:
        topx,topy = mover.find_top_left(tgo,cam_short,center)
        return False
    except gc.EdgeNotFound:
        return True
    except:
        return False

def edge_detection_test(img):
    tup_output = gc.is_edge_in_center(img,darkness_thresh = 225, thresh_val= 150)
    return tup_output
def edge_detection_try_all_files_test():
    path_to_edge_images = Path(glob.glob(str(root)+"/**/*calibrating_edge_detection",recursive = True)[0])
    list_of_imgs_paths = os.listdir(str(path_to_edge_images))
    expected_digits=[8,4,8,5]
    for i,img_path in enumerate(list_of_imgs_paths):
        img = cv2.imread(str(path_to_edge_images / img_path))
        result = edge_detection_test(img)
        percent = gc.get_percent_of_img(img, x = True,DEBUG=True)
        if expected_digits[i] != result:
            return False
    return True
def get_percent_of_img_test_x(image):
    percent = gc.get_percent_of_img(image, x = True,DEBUG=True)
    return percent
def get_percent_of_img_test_y(image):
    percent = gc.get_percent_of_img(image, x = False,DEBUG=True)
    return percent
def get_percent_of_img_try_all_files_test():
    path_to_edge_images = Path(glob.glob(str(root)+"/**/*calibrating_edge_detection",recursive = True)[0])
    list_of_imgs_paths = os.listdir(str(path_to_edge_images))
    expected_digits_x =[0.812962962962963,
                        0.8074074074074075,
                        0.9050925925925926,
                        0.7212962962962963,
                        ]
    expected_digits_y =[0.812962962962963,
                        0.8074074074074075,
                        0.9050925925925926,
                        0.7212962962962963,
                        ]
    for i,img_path in enumerate(list_of_imgs_paths):
        img = cv2.imread(str(path_to_edge_images / img_path))
        percent_x = get_percent_of_img_test_x(img)
        percent_y = get_percent_of_img_test_x(img)
        if abs(expected_digits_x[i] - percent_x) >0.001:
            return False
        if abs(expected_digits_y[i] - percent_y) >0.001:
            return False
    return True
if __name__ =="__main__":
    print("basic run test ")
    # basic_run_test()
    # # result = #find_top_left_test()
    find_general_loc_test()
    readImg_test()
    result = get_percent_of_img_try_all_files_test()
    # print("result:", result)
