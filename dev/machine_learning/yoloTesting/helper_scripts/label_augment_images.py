import cv2
import numpy as np
import os
import sys
import shutil
from pathlib import Path
import random as rd
import glob
ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in range(100):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
root = get_root()
path_to_yolo = Path(glob.glob(str(root)+"/**/*Nano_yolo.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_yolo))
import Nano_yolo as NY

path_to_yolo = Path(glob.glob(str(root)+"/**/*image_methods.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_yolo))
import image_methods as wrp


path_to_classes = path_to_yolo / "new_class_spec.yaml"

path_to_yolo = Path(glob.glob(str(root)+"/**/*KNNAlgo.py",recursive = True)[0]).parent.absolute()
path_to_img_folder = path_to_yolo / "Labeling" /"images_raw"
path_to_label_folder = path_to_yolo / "Labeling" /"labels_raw"



def draw_labels(img,data,colors):
    out_img = img.copy()
    im_h,im_w,d = out_img.shape
    class_name = data[0]["Class"]
    class_num = 0
    for point in data:
        if (class_name != point["Class"]):
            class_num +=1
            class_name = point["Class"]
        h = int(im_h*point["height"])
        w = int(im_w*point["width"])
        x = int((im_w*point["Xcenter"])-w/2)
        y = int((im_h*point["Ycenter"])-h/2)
        cv2.rectangle(out_img, (x, y), (w+x,h+y), (int(colors[class_num][0]),int(colors[class_num][1]),int(colors[class_num][2])), 5)
    return out_img
def read_file_as_dic(infileLoc,classes):
    data = []
    with open(infileLoc) as file:
        for line in file:
            stringarray = line[:-1].split(" ")
            point = {}
            point["Class"] = classes[int(stringarray[0])]
            point["Xcenter"] = (float(stringarray[1]))
            point["Ycenter"] = (float(stringarray[2]))
            point["width"] = (float(stringarray[3]))
            point["height"] = (float(stringarray[4]))
            data.append(point)

    new_dic_list = sorted(data, key = lambda i: i['Class'])
    return new_dic_list
def augment_noise(label_path,img_path,augmented_fraction = 0.15):
        imgs = os.listdir(img_path)
        imgs.sort()
        print(imgs)
        # max_name = int(imgs[-1][:imgs[-1].find(".")])+1
        # rd.shuffle(imgs)
        save_path_img = Path(img_path).parent.absolute() /"added_noise_img"
        save_path_labels = Path(label_path).parent.absolute() /"added_noise_labels"
        label_path = Path(label_path)



        try:
            os.listdir(str(save_path_img))
        except:
            try:
                os.mkdir(str(save_path_img))
            except:
                print("failure")
        try:
            os.listdir(str(save_path_labels))
        except:
            try:
                os.mkdir(str(save_path_labels))
            except:
                print("failure")
        for n,i in enumerate(imgs[:int(len(imgs)*augmented_fraction)]):
            label_name = i[:i.find(".")] +".txt"
            name =  n
            img = cv2.imread(str(Path(img_path)/i))
            blured = wrp.blur(img,amt=rd.randint(1, 5))
            with_noise = wrp.add_noise(blured, amt=rd.randint(0, 3), types= rd.randint(1, 4))
            distorted = wrp.randomGradientLight(with_noise, rd.randint(0, 3))
            cv2.imwrite(str(save_path_img / f"a{name}.png"),distorted)
            shutil.copyfile(str(label_path/label_name), str(save_path_labels/ f"a{name}.txt"))

def read_show_folder(label_path,img_path,classes):

    colors = [list(np.random.choice(range(255),size=3)) for i in classes]
    imgs = os.listdir(img_path)
    for i in imgs:
        name = i[:i.find(".")]
        label_data = read_file_as_dic(str(Path(label_path)/f"{name}.txt"),classes)
        img = cv2.imread(str(Path(img_path)/i))
        out = draw_labels(img,label_data,colors)
        cv2.imshow("original",img)
        cv2.imshow("boxed",out)
        cv2.waitKey(500)
if __name__ == "__main__":

    classes = NY.read_yaml_class_names(str(path_to_classes))
    # read_show_folder(str(path_to_label_folder),str(path_to_img_folder),classes)
    PathToFolder = root/"dev"/"machine_learning"/"yoloTesting"/"Labeling"/"images_raw"
    PathToFolderL = root/"dev"/"machine_learning"/"yoloTesting"/"Labeling"/"labels_raw"
    # augment_noise(PathToFolderL,PathToFolder,augmented_fraction = 0.5)
    # count = 0
    # for i in range(10):
    #     PathToFolder = root/"dev"/"machine_learning"/"yoloTesting"/"Labeling"/"images_raw"/str(i)
    #     files = os.listdir(PathToFolder)
    #     # for file in files:
    #     #     count +=1
    #     #     shutil.copyfile(str(root/"dev"/"machine_learning"/"numberRecognition"/"DigitsGreen"/f"{i}"/str(file)), str(root/"dev"/"machine_learning"/"numberRecognition"/"DigitsGreen"/f"{i}"/f"{count}.png"))


    read_show_folder(str(path_to_label_folder.parent.absolute() /"added_noise_labels"),str(path_to_img_folder.parent.absolute() /"added_noise_img"),classes)

    # Z = 3 + 1.2 = 4.2
