import os
import sys
path ="/home/pequode/Desktop/github/NanoView_G33/dev/machine_learning/yoloTesting/raw_images/"
files = os.listdir(path)
for i,file in enumerate(files):
    print(i,file)
    os.rename(path+file,path+str(i)+".jpg")
