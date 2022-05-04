import os
import random
import shutil
import glob
import sys
from pathlib import Path
import cv2


ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split("/"):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
root = get_root()

path_to_img_meth = Path(glob.glob(str(root)+"/**/*image_methods.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_img_meth))
import image_methods as wrp

path_to_ones = Path(glob.glob(str(root)+"/**/*DigitsGreen",recursive = True)[0])/"1"
print(str(path_to_ones))
ones = os.listdir(str(path_to_ones))
print(ones)
for im in ones:
    img = cv2.imread(str(path_to_ones/im))
    # greater_dim = img.shape[0] if img.shape[0]>img.shape[1] else img.shape[1]
    # smaller_dim = img.shape[0] if img.shape[0]<img.shape[1] else img.shape[1]
    # canv = wrp.makeCanvas(greater_dim+1, greater_dim+1, src="background.png")
    # reult,stuf = wrp.overlay_image_alpha(canv,cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) , 10, 0)#cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # cv2.imshow("sd",)
    # cv2.waitKey(0)
    cv2.imwrite(str(path_to_ones / im), cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
