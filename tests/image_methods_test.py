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
path_to_img_meth = Path(glob.glob(str(root)+"/**/*image_methods.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_img_meth))
import image_methods as wrp

def makeCan_test():
    canv = wrp.makeCanvas(500,500)
    canv1 = wrp.makeCanvas(100,100,src = "img_test_merge.png")
    img = cv2.imread("img_test_merge.png")
    canv2 = wrp.makeCanvas(1000,1000,img = img)
    cv2.imshow("c1",canv)
    cv2.imshow("c2",canv1)
    cv2.imshow("c3",canv2)
    cv2.waitKey(5000)
    return True
if __name__ =="__main__":
    print("basic run test ")
    result = makeCan_test()
    print("result:", result)
