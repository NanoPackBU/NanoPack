import os
import random
import shutil
import glob
import sys
from pathlib import Path

ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split("/"):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
root = get_root()

pathLabeling = Path(glob.glob(str(root)+"/**/*Labeling",recursive = True)[0])
pathTrainingData =Path(glob.glob(str(root)+"/**/*training_data",recursive = True)[0])
trainamt=0.8
valamt = 0.2

files = os.listdir(str(pathLabeling/"images"))

random.shuffle(files)
assert(trainamt+valamt==1)
trainFnames = files[:int(trainamt*len(files))]
valFnames = files[int(trainamt*len(files)):]
print(len(valFnames),"+",len(trainFnames),"=",len(valFnames)+len(trainFnames),"==",len(files))
for name in trainFnames:
    baseName = name[:name.find(".")]
    shutil.copyfile(str(pathLabeling/"images"/ str(baseName+".jpg")), str(pathTrainingData /"images"/"train"/str(baseName+".jpg")))
    shutil.copyfile(str(pathLabeling/"labels"/ str(baseName+".txt")),  str(pathTrainingData /"labels"/"train"/str(baseName+".txt")))
for name in valFnames:
    baseName = name[:name.find(".")]
    shutil.copyfile(str(pathLabeling/"images"/ str(baseName+".jpg")), str(pathTrainingData /"images"/"val"/str(baseName+".jpg")))
    shutil.copyfile(str(pathLabeling/"labels"/ str(baseName+".txt")), str(pathTrainingData /"labels"/"val"/str(baseName+".txt")))
