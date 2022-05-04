
import sys
import cv2
from pathlib import Path
import os
import glob
## used to get the pathing sorted
##TODO: test card
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
# gets the src and dest folders
ml_folder = root /"dev"/"machine_learning"/"yoloTesting"
srcpath = ml_folder/"raw_images"
destpath = ml_folder/ "Labeling"
files = os.listdir(srcpath)
# for files in in dir format and then write
for i,file in enumerate(files):
    print(i,file)
    img = cv2.imread(str(srcpath/file))
    white_balenced = wrp.white_balance(img)
    cv2.imwrite(str(destpath/file),white_balenced)
