# from doctest import OutputChecker
# import sys
# import cv2
# import yaml
# from pathlib import Path
# import glob

# ROOT = ".git"


# def get_root():
#     new_path = Path(__file__)
#     for i in str(new_path).split("/"):
#         if len(glob.glob(str(new_path / ROOT))) > 0:
#             break
#         new_path = new_path.parent.absolute()
#     return new_path


# root = get_root()

# path_to_detect = Path(glob.glob(str(root) + "/**/*detectMOD.py", recursive=True)[0]).parent.absolute()
# sys.path.insert(1, str(path_to_detect))
# import detectMOD

# path_to_img_meth = Path(glob.glob(str(root) + "/**/*image_methods.py", recursive=True)[0]).parent.absolute()
# sys.path.insert(1, str(path_to_detect))
# import image_methods as wrp


# class CustomYolo:
#     def __init__(self, config_path="config_distance.yml", DEBUG=True, saveDir="none"):
#         self.config = self.readConfig(config_path)
#         # assumes that the weights and classes are in the same folder as this file
#         program_path = Path(__file__).parent.absolute()
#         name_class_definition = str(program_path / self.config["pathing"]["machine_learning"]["object_detection"]["class_defs"])
#         name_weight_file = str(program_path / self.config["pathing"]["machine_learning"]["object_detection"]["weights"])
#         self.class_definition  = str(Path(glob.glob(str(root) + "/**/*"+name_class_definition, recursive=True)[0]))
#         self.weight_path = str(Path(glob.glob(str(root) + "/**/*"+name_weight_file, recursive=True)[0]))
#         self.DEBUG = DEBUG
#         self.save_dir = saveDir
#         if self.DEBUG:
#             print(f"Intialized:\n\tconfig dir:{self.config}\n\class def:{self.class_definition}\n\t")

#     def readConfig(self, filename="config_distance.yml"):
#         data = {}
#         with open(filename, "r") as stream:
#             try:
#                 data = yaml.safe_load(stream)
#             except yaml.YAMLError as exc:
#                 print(exc)
#                 raise
#         return data

#     def formate_image_get_mm(self, img):
#         right_round = cv2.rotate(img, cv2.ROTATE_180)
#         white_shift = wrp.white_balance(right_round)
#         croped = wrp.crop(white_shift,
#                           self.config["camera"][0]["image"]["crop"]["top"],
#                           self.config["camera"][0]["image"]["crop"]["bottom"],
#                           self.config["camera"][0]["image"]["crop"]["left"],
#                           self.config["camera"][0]["image"]["crop"]["right"])
#         out = self.get_coordinates_mm(croped)
#         return out

#     def get_coordinates_mm(self, img):
#         print("HERE")
#         bedwidth = self.config["physical"]["traveler_box"]["x"] + self.config["physical"]["clam_shell_box"]["x"]
#         beddepth = self.config["physical"]["traveler_box"]["y"]
#         cos = detectMOD.run(GKSPERAM=img,
#                             weights=self.weight_path,
#                             source=str(Path(__file__).parent.absolute() / "ml_guess.jpg"),  # try to phase out
#                             data=self.class_definition,
#                             imgsz=[512, 512],
#                             conf_thres=self.config["pathing"]["machine_learning"]["object_detection"][
#                                 "confidence_thresh"],  # confidence threshold
#                             save_txt=(self.save_dir != "none"),
#                             project=str(Path(__file__).parent.absolute() / self.save_dir),
#                             nosave=False,
#                             device='cpu'
#                             )

#         dicOfCos = convert_out_tensor_to_dic(cos, self.class_definition, self.DEBUG)
#         merge_too_close(dicOfCos)
#         if self.DEBUG:
#             drawData(dicOfCos, img)
#         dicOFMM = convertFileToMmCos(dicOfCos, bedwidthmm=bedwidth,
#                                      beddepthmm=beddepth,
#                                      cameraoffset_y=self.config["machine"]["distance_to_center_cam"]["y"],
#                                      cameraoffset_x=self.config["machine"]["distance_to_center_cam"]["x"])
#         return dicOFMM


# def merg_if_overlapping(cx, cy, w, h, cx1, cy1, w1, h1, x_padding=0.05, ):
#     width1 = x_padding * 2 + w
#     y = cy - h / 2
#     y1 = cy1 - h1 / 2

#     x = cx - w / 2
#     x1 = cx1 - w1 / 2

#     diff = abs(y1 - y)
#     if diff > h:
#         return False, 0
#     xe = x + w
#     x1e = x1 + w1
#     connected = (x < x1 < xe) or (x1 < x < x1e)
#     if not connected:
#         return False, 0
#     x_out = x if x < x1 else x1
#     xe_out = xe if xe > x1e else e1x
#     y_out = y if y < y1 else y1
#     ye_out = y if y > y1 else y1
#     w = xe_out - x_out
#     h = ye_out - y_out
#     x = x + w / 2
#     y = y + h / 2
#     return x, y, w, h


# # def merge_too_close(dic_list):
# #     new_list = []
# #     clamshells = []

#     for item in dic_list:
#         if item["Class"] == "Clamshell":
#             clamshells.append(item)
#             print("HERE THEY ARE:", item)
#         else:
#             new_list.append(item)
#             print("kalay")
#     # check for intersecting
#     new_clamshell_list = clamshells
#     for item in clamshells:
#         for i in new_clamshell_list:
#             if i["Xcenter"] == item["Xcenter"] and i["Ycenter"] == item["Ycenter"]:
#                 continue

#             x, y, w, h = merg_if_overlapping(item["Xcenter"], item["Ycenter"], item["Width"], item["Hieght"],
#                                              i["Xcenter"], i["Ycenter"], i["Width"], i["Hieght"])
#             dic["Class"] = "Clamshell"
#             dic["Xcenter"] = x
#             dic["Ycenter"] = y
#             dic["Width"] = w
#             dic["Hieght"] = h
#             dic["conf"] = item["conf"]
#             new_clamshell_list.append(dic)
#             new_clamshell_list.remove(i)
#             break


# def read_yaml_class_names(path_to_class, debug=False):
#     if (debug): print("CLASSS SPEC PATH ______?", path_to_class)
#     with open(path_to_class) as file:
#         # The FullLoader parameter handles the conversion from YAML
#         # scalar values to Python the dictionary format
#         classlist = yaml.full_load(file)
#     classes = classlist["names"]
#     print(classes)
#     return classes


# def read_file_as_dic(infileLoc, path_to_class):
#     data = []
#     classes = read_yaml_class_names(path_to_class)
#     with open(infileLoc) as file:
#         for line in file:
#             stringarray = line[:-1].split(" ")
#             point = {}
#             print(stringarray)
#             point["Class"] = classes[int(stringarray[0])]
#             point["Xcenter"] = (float(stringarray[1]))
#             point["Ycenter"] = (float(stringarray[2]))
#             point["conf"] = float(stringarray[5])
#             data.append(point)
#     return data


# def convert_out_tensor_to_dic(tens, path_to_class, debug=False):
#     new_dic_list = []
#     print("BABAYAGA\n\n\n")
#     classes = read_yaml_class_names(path_to_class, debug)
#     for line in tens:
#         dic = {}
#         if (debug): print("LINE:", line)
#         dic["Class"] = classes[int(line[0])]
#         dic["Xcenter"] = float(line[1])
#         dic["Ycenter"] = 1 - float(line[2])
#         dic["Width"] = float(line[3])
#         dic["Hieght"] = float(line[4])
#         dic["conf"] = float(line[5])
#         new_dic_list.append(dic)
#     # sort dic so that traveller is last
#     new_dic_list = sorted(new_dic_list, key=lambda i: i['Class'])
#     if (debug): print(new_dic_list)
#     return new_dic_list


# def convertFileToMmCos(dicL, bedwidthmm=100, beddepthmm=100, cameraoffset_x=0, cameraoffset_y=0):
#     new_dic_list = []
#     for dic in dicL:
#         dic["Xcenter"] = dic["Xcenter"] * bedwidthmm - cameraoffset_x
#         dic["Ycenter"] = dic["Ycenter"] * beddepthmm - cameraoffset_y
#         dic["Width"] = dic["Width"] * bedwidthmm
#         dic["Hieght"] = dic["Hieght"] * beddepthmm
#         new_dic_list.append(dic)
#     return new_dic_list


# def drawData(dic, img):
#     dy = img.shape[0]
#     dx = img.shape[1]
#     img = cv2.resize(img, (dx, dy))
#     for i in dic:
#         img = cv2.circle(img, (int(i["Xcenter"] * dx), int(i["Ycenter"] * dy)), 1, (0, 0, 255), -1)
#     cv2.imshow("testUnitConversion", img)
#     cv2.waitKey(1000)
