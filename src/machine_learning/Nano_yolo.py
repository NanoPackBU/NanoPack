import cv2
import yaml
import sys
import glob
from pathlib import Path
import time
# custom



pathChar = "/"
isWindows = sys.platform.startswith('win')
if (isWindows): pathChar = "\\"
ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in range(100):
        if (len(glob.glob(str(new_path / ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path

root = get_root()
pathToSrc = Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0]).parent.absolute()

path_to_detect = Path(glob.glob(str(root) + "/**/*detectMOD.py", recursive=True)[0]).parent.absolute()
sys.path.insert(1, str(path_to_detect))
import detectMOD

path_to_img_meth = Path(glob.glob(str(root) + "/**/*image_methods.py", recursive=True)[0]).parent.absolute()
print(path_to_img_meth)
sys.path.insert(1, str(path_to_img_meth))
import image_methods as im


def readConfig(filename=str(pathToSrc) + pathChar + "config_distance.yml"):
    data = {}
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return data


class CustomYolo:
    def __init__(self, config_path=str(pathToSrc) + pathChar + "config_distance.yml", DEBUG=True, saveDir="none"):
        self.config = readConfig(config_path)
        # assumes that the weights and classes are in the same folder as this file
        program_path = Path(__file__).parent.absolute()
        name_class_definition = self.config["pathing"]["machine_learning"]["object_detection"]["class_defs"]
        name_weight_file = self.config["pathing"]["machine_learning"]["object_detection"]["weights"]
        print(name_weight_file,name_class_definition)
        self.class_definition  = str(Path(glob.glob(str(root) + "/**/*"+name_class_definition, recursive=True)[0]))
        self.weight_path = str(Path(glob.glob(str(root) + "/**/*"+name_weight_file, recursive=True)[0]))
        self.DEBUG = DEBUG
        self.save_dir = saveDir
        if self.DEBUG:
            print(f"Initialized:\n\tconfig dir:{self.config}\n\class def:{self.class_definition}\n\t")

    def formate_image_get_mm(self, img):
        # right_round = cv2.rotate(img, cv2.ROTATE_180)
        white_shift = im.white_balance(img)
        cropped = im.crop(white_shift,
                          self.config["camera"][0]["image"]["crop"]["top"],
                          self.config["camera"][0]["image"]["crop"]["bottom"],
                          self.config["camera"][0]["image"]["crop"]["left"],
                          self.config["camera"][0]["image"]["crop"]["right"])
        out = self.get_coordinates_mm(cropped)
        return out

    def get_coordinates_mm(self, img):
        bed_width = self.config["physical"]["traveler_box"]["x"] + self.config["physical"]["clam_shell_box"]["x"]
        bed_depth = self.config["physical"]["traveler_box"]["y"]
        cos = detectMOD.run(GKSPERAM=img,
                            weights=self.weight_path,
                            source=str(Path(__file__).parent.absolute() / "ml_guess.jpg"),  # try to phase out
                            data=self.class_definition,
                            imgsz=[512, 512],
                            conf_thres=self.config["pathing"]["machine_learning"]["object_detection"][
                                "confidence_thresh"],  # confidence threshold
                            save_txt=(self.save_dir != "none"),
                            project=str(Path(__file__).parent.absolute() / self.save_dir),
                            nosave=False,
                            device='cpu'
                            )

        dicOfCos = convert_out_tensor_to_dic(cos, self.class_definition, self.DEBUG)
        # merge_too_close(dicOfCos)
        if self.DEBUG:
            drawData(dicOfCos, img)
        dicOFMM = convertFileToMmCos(dicOfCos, bedwidthmm=bed_width,
                                     beddepthmm=bed_depth,
                                     cameraoffset_y=self.config["machine"]["distance_to_center_cam"]["y"],
                                     cameraoffset_x=self.config["machine"]["distance_to_center_cam"]["x"])
        return dicOFMM


def merge_if_overlapping(cx, cy, w, h,
                        cx1, cy1, w1, h1,
                        x_padding=0.05,DEBUG = False,img = None):
    #convert center to edge box description
    y = cy - (h / 2)
    y1 = cy1 - (h1 / 2)

    x = cx - (w*(1+(x_padding)) / 2)
    x1 = cx1 - (w1*(1+(x_padding)) / 2)

    # check for intersections
    overlapping = im.Intersecting(x, y, w*(1+(2*x_padding)), h,
                                   x1, y1, w1*(1+(2*x_padding)), h1)
    if not (overlapping):
        if (DEBUG):
            canv = None
            if(type(img) == type(None)):
                canv = im.makeCanvas(500,500)
            else:
                print("image in")
                canv = im.makeCanvas(500,500,img = img)
            cv2.rectangle(canv, (int(x), int(y)), (int(x+w*(1+(x_padding))),int(y+h)), (0,0,255), 2)
            cv2.rectangle( canv, (int(x1), int(y1)), (int(x1+w1*(1+(x_padding))),int(y1+h1)), (0,255,0), 2)
            cv2.imshow("new vs old",canv)
            cv2.waitKey(1000)
        return False, 0,0,0,0

    xe = x + (w*(1+(x_padding)))
    x1e = x1 + (w1*(1+(2*x_padding)))

    ye = y+ h
    y1e = y1+h1

    x_out = x if x < x1 else x1
    xe_out = xe if xe > x1e else x1e
    y_out = y if y < y1 else y1
    ye_out = ye if ye > y1e else y1e
    wo = xe_out - x_out
    ho = ye_out - y_out
    cxo = x + w / 2
    cyo = y + h / 2
    if (DEBUG):
        canv = None

        if(type(img) == type(None)):
            print("no image in")
            canv = im.makeCanvas(500,500)
        else:
            print("image in")
            canv = im.makeCanvas(500,500,img = img)
        cv2.rectangle(canv, (int(x_out), int(y_out)), (int(x_out+wo),int(y_out+ho)), (255,0,0), 5)
        cv2.rectangle(canv, (int(x), int(y)), (int(x+w*(1+(x_padding))),int(y+h)), (0,0,255), 2)
        cv2.rectangle( canv, (int(x1), int(y1)), (int(x1+w1*(1+(x_padding))),int(y1+h1)), (0,255,0), 2)
        cv2.imshow("new vs old",canv)
        cv2.waitKey(1000)
    return True,cxo, cyo, w, h


def merge_too_close(dic_list,DEBUG = False,img = None):
    new_list = []
    clamshells = []

    for item in dic_list:
        if item["Class"] == "Clamshell":
            clamshells.append(item)
            if DEBUG: print("HERE THEY ARE:", item)
        else:
            new_list.append(item)

    # check for intersecting
    new_clamshell_list = clamshells
    for item in clamshells:
        for i in new_clamshell_list:
            if i["X_center"] == item["X_center"] and i["Y_center"] == item["Y_center"]:
                if DEBUG:print("skipping self")
                continue
            if DEBUG: print(item, "vs" ,i)
            output = merge_if_overlapping(item["X_center"], item["Y_center"], item["Width"], item["Height"],
                                              i["X_center"], i["Y_center"], i["Width"], i["Height"],x_padding=0.01,DEBUG = False)
            if(not output[0]):
               if DEBUG: print("no merge")
               continue
            if DEBUG: print("merge!")
            new_clamshell_list.remove(item)
            new_clamshell_list.remove(i)
            ret,x, y, w, h = output
            dic = i
            dic["Class"] = "Clamshell"
            dic["X_center"] = x
            dic["Y_center"] = y
            dic["Width"] = w
            dic["Height"] = h
            dic["conf"] = item["conf"]
            new_clamshell_list.append(dic)
            new_clamshell_list.remove(i)
            break
    new_list = new_list+new_clamshell_list
    if DEBUG: print(len(new_list))
    if DEBUG: print(new_list)
    if (DEBUG):
        canv = None

        if(type(img) == type(None)):
            print("no image in")
            canv = im.makeCanvas(500,500)
        else:
            print("image in")
            canv = im.makeCanvas(500,500,img = img)
        for item in new_list:
                h = item["Height"]
                w = item["Width"]
                x_out = item["X_center"] - (w/2)
                y_out = item["Y_center"] - (h/2)

                cv2.rectangle(canv, (int(x_out), int(y_out)), (int(x_out+w),int(y_out+h)), (0,0,255), 5)
        for item in dic_list:
                h = item["Height"]
                w = item["Width"]
                x_out = item["X_center"] - (w/2)
                y_out = item["Y_center"] - (h/2)

                cv2.rectangle(canv, (int(x_out), int(y_out)), (int(x_out+w),int(y_out+h)), (0,255,0), 2)
        cv2.imshow("new vs old",canv)
        cv2.waitKey(1000)
    return new_list

def read_yaml_class_names(path_to_class, debug=False):
    if (debug): print("CLASSS SPEC PATH ______?", path_to_class)
    with open(path_to_class) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        classlist = yaml.full_load(file)
    classes = classlist["names"]
    print(classes)
    return classes


def read_file_as_dic(infileLoc, path_to_class):
    data = []
    classes = read_yaml_class_names(path_to_class)
    with open(infileLoc) as file:
        for line in file:
            stringarray = line[:-1].split(" ")
            point = {}
            print(stringarray)
            point["Class"] = classes[int(stringarray[0])]
            point["X_center"] = (float(stringarray[1]))
            point["Y_center"] = (float(stringarray[2]))
            point["conf"] = float(stringarray[5])
            data.append(point)
    return data


def convert_out_tensor_to_dic(tens, path_to_class, debug=False):
    new_dic_list = []

    classes = read_yaml_class_names(path_to_class, debug)
    for line in tens:

        dic = {}
        if debug:
            print("LINE:", line)
        dic["Class"] = classes[int(line[0])]
        dic["X_center"] = 1- float(line[1])
        dic["Y_center"] = float(line[2])#1 - float(line[2])#
        dic["Width"] = float(line[3])
        dic["Height"] = float(line[4])
        dic["conf"] = float(line[5])
        new_dic_list.append(dic)
    # sort dic so that traveller is last
    new_dic_list = sorted(new_dic_list, key=lambda i: i['Class'])
    if debug:
        print(new_dic_list)
    return new_dic_list


def convertFileToMmCos(dicL, bedwidthmm=100, beddepthmm=100, cameraoffset_x=0, cameraoffset_y=0):
    new_dic_list = []
    for dic in dicL:
        dic["X_center"] = dic["X_center"] * bedwidthmm - cameraoffset_x
        dic["Y_center"] = dic["Y_center"] * beddepthmm - cameraoffset_y
        dic["Width"] = dic["Width"] * bedwidthmm
        dic["Height"] = dic["Height"] * beddepthmm
        new_dic_list.append(dic)
    return new_dic_list


def drawData(dic, img):
    dy = img.shape[0]
    dx = img.shape[1]
    img = cv2.resize(img, (dx, dy))
    for i in dic:
        img = cv2.circle(img, (int(i["X_center"] * dx), int((1-i["Y_center"]) * dy)), 5, (0, 0, 255), -1)
    cv2.imshow("testUnitConversion", img)
    cv2.waitKey(1000)
