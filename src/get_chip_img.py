import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import machine_learning.image_methods as wrp
import machine_learning.yolo_ml.yolov5
import tinyg.tinyg
# custom libraries
from machine_learning.Nano_yolo import CustomYolo


# Error States
class TravelerNotFound(Exception):
    pass


class ClamshellsNotFound(Exception):
    pass


class EdgeNotFound(Exception):
    pass

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

class chip_imgr():
    def __init__(self, config_path=str(pathToSrc) + pathChar + "config_distance.yml", DEBUG=True):
        if DEBUG:
            print("defined chip imgr class")
        self.object_detection = CustomYolo(config_path=config_path, saveDir="runs", DEBUG=True)
        self.config = self.object_detection.config  # set the config object to use the same one as the sub class
        # scale mm to tinyg units
        self.Y_FAC = self.config["physical"]["tinyg_units_conversion_fac"]["y"]
        self.X_FAC = self.config["physical"]["tinyg_units_conversion_fac"]["x"]
        self.Z_FAC = self.config["physical"]["tinyg_units_conversion_fac"]["z"]
        self.DEBUG = DEBUG

    # camera method
    # Starts the two cameras
    def start_cam(self):
        wide = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        short = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        short.set(cv2.CAP_PROP_BUFFERSIZE, 0)  # Makes the capture buffer have a size of zero
        wide.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        return wide, short

    # camera method
    # deal with buffering of images in cv2
    def readImg(self, cam, tp, iters=0, reticle=False):
        fps = 7.5
        step = (1 / fps)
        if iters == 0:
            iters = int(tp / step)
            _, img = cam.read()
            for _ in range(iters):
                _, img = cam.read()
                time.sleep(step)
        else:
            _, img = cam.read()
            for _ in range(iters):
                _, img = cam.read()

        if reticle:
            # Getting the height and width of the image
            height = img.shape[0]
            width = img.shape[1]

            # Drawing the lines
            cv2.line(img, (int(width / 2), 0), (int(width / 2), height), (0, 0, 255), 2)
            cv2.line(img, (0, int(height / 2)), (width, int(height / 2)), (0, 0, 255), 2)
        return img

    # ml method
    # finds classes with a cam
    def find_general_loc(self, cap_wide, attempts=3):  # trys three times to take an image of the bed an extract data
        # offsets of image from bed
        startbox_x = self.config["machine"]["work_area_offset"]["x"]
        startbox_y = self.config["machine"]["work_area_offset"]["y"]
        for i in range(attempts):
            # _,imgTaken = cap_wide.read()
            imgTaken = self.readImg(cap_wide, 0.2, iters=24)
            # print(imgTaken.shape)
            # perform yolo
            coors = self.object_detection.formate_image_get_mm(imgTaken)
            if len(coors) > 0:  # it found something
                travelerLoc = [d for d in coors if d['Class'] == 'Traveler']
                # if incorrect # travelers
                if len(travelerLoc) != 1:
                    print("issues with travelers")
                    continue
                x_of_traveler_in_bed = travelerLoc[0]["X_center"] + startbox_x
                y_of_traveler_in_bed = travelerLoc[0]["Y_center"] + startbox_y
                clamshellLocs = [
                    {'Class': d['Class'],
                     "X_center": d['X_center'] + startbox_x,
                     "Y_center": d["Y_center"] + startbox_y,
                     "conf": d["conf"],
                     } for d in coors if d['Class'] == 'Clamshell']
                # if no clamshells
                if not len(clamshellLocs):
                    print("no clams found")
                    continue
                return x_of_traveler_in_bed, y_of_traveler_in_bed, clamshellLocs
        else:
            if not len(clamshellLocs):
                raise ClamshellsNotFound
        print("no trav found")
        raise TravelerNotFound

    # computer vision method
    # move over clamshell/traveler till edge found
    def find_edge(self, tgo, cam_short, center, clam=False, attemps=3):

        ## Move to center
        tgo.MoveLinear(300, center[0] * self.X_FAC, center[1] * self.Y_FAC, 0, 0)

        ## Take image of current position and make sure edge is in view
        iters = 24
        imgTaken = self.readImg(cam_short, 0.2, iters=iters)
        if 0 == is_edge_in_center(imgTaken, DEBUG=self.DEBUG):
            for i in range(attemps):
                imgTaken = self.readImg(cam_short, 0.2, iters=iters)
                if 0 != is_edge_in_center(imgTaken, DEBUG=self.DEBUG):
                    break
            else:
                raise EdgeNotFound("not in container")  # make sure camera is in container

        ## Scan over the traveler for edge
        percent_edge_of_img = 0
        corner_offset_amt = -1
        step_size = self.config["algo"]["edge_detection"]["step_size"]
        side = 0
        if(clam):
            list_of_looks = range(int(self.config["physical"]["clam_shell"]["x_open"] / step_size) + 2)
        else:
            list_of_looks = range(int(self.config["physical"]["traveler"]["y"] / step_size)+5)
        for i in list_of_looks:
            imgTaken = self.readImg(cam_short, 0.2, iters=iters)
            if self.DEBUG:
                cv2.imshow("cap", imgTaken)
                cv2.waitKey(10 * iters)
            side = is_edge_in_center(imgTaken, DEBUG=self.DEBUG)

            ## If edge is not in center, find percent
            upper_bound = 6 if clam else 8
            lower_bound = 4 if clam else 2
            if side in [lower_bound, upper_bound]:
                corner_offset_amt = i-1 # its current coordinates are the index we are measuring minus one
                percent_edge_of_img = get_percent_of_img(imgTaken, x=clam, window_percent=0.8, search_perc=0.005,
                                                           DEBUG=self.DEBUG)
                break

            if (clam):
                tgo.MoveLinear(300, (center[0] - (i * step_size)) * self.X_FAC, center[1] * self.Y_FAC, None, None)
            else:
                tgo.MoveLinear(300, (center[0]) * self.X_FAC, (center[1] + (i * step_size)) * self.Y_FAC, None, None)
        else:
            raise EdgeNotFound("Edge not found!")

        if (clam):
            mm_to_move = ((0.5 - percent_edge_of_img) * self.config["camera"][1]["width"])
            found_edge = center[0] - (corner_offset_amt * step_size) - mm_to_move
        else:
            mm_to_move = ((0.5 - percent_edge_of_img) * self.config["camera"][1]["height"])
            found_edge = (corner_offset_amt * step_size) + center[1] + mm_to_move

        if self.DEBUG:
            print(f"PECENT OF THE EDGE FROM TOP: {percent_edge_of_img}")
            print(f"PERCENT FOR MOVEMENT {(0.5 - percent_edge_of_img)}")
            print(f"MM TO MOVE {mm_to_move}")
            print(f"CURR FOUND EDGE {found_edge - mm_to_move}")
            print(f"NEW FOUND EDGE {found_edge}")

        return found_edge

    # move method
    # moves to the top left corner of the clamshell using edge detection, given point within it
    def find_top_clamshell(self, tgo, cam_short, center, constrain_row=1):
        left_edge = self.find_edge(tgo, cam_short, center, clam=True)
        top_edge = self.config["physical"]["constraints"]["clamshell_r" + str(constrain_row) + "_top"]

        if self.DEBUG:
            tgo.MoveLinear(300, left_edge * self.X_FAC,
                                top_edge * self.Y_FAC,
                                None, None)
            img = self.readImg(cam_short, 0.2, iters=24, reticle=True)
            cv2.imshow("loss", img)
            cv2.waitKey(500)
        return left_edge, top_edge

    # move method
    # moves to the top left corner of the traveler using edge detection, given point within it
    def find_top_traveler(self, tgo, cam_short, center):  # enter the serial obj and the center tuple in mm
        top_edge = self.find_edge(tgo, cam_short, center)
        left_edge = self.config["physical"]["constraints"]["traveler"]["x"]

        if self.DEBUG:
            tgo.MoveLinear(300, left_edge * self.X_FAC,
                                top_edge * self.Y_FAC,
                                None, None)
            img = self.readImg(cam_short, 0.2, iters=24, reticle=True)
            cv2.imshow("loss", img)
            cv2.waitKey(500)
        return left_edge, top_edge

    # math method
    # returns the location of the start of the top-left chip from the top left corner of clamshell
    def clamshell_chip_start(self, left_edge, top_edge):
        x_offset = self.config["physical"]["clam_shell"]["inner"]["x_offset"]
        y_offset = self.config["physical"]["clam_shell"]["inner"]["y_offset"]
        topx = left_edge + x_offset
        topy = top_edge - y_offset
        return topx, topy

    # math method
    # returns the location of the start of the top-left chip from the top left corner of traveler
    def traveller_chip_start(self, left_edge, top_edge):
        x_offset = self.config["physical"]["traveler"]["inner"]["x_offset"]
        y_offset = self.config["physical"]["traveler"]["inner"]["y_offset"]
        topx = left_edge + x_offset
        topy = top_edge - y_offset
        return topx, topy

    # move method
    # given traveler top-left chip x,y move to a given chip num
    def move_cam_to_chip_in_trav(self, tgo, t_top_x, t_top_y, num):
        x_coor, y_coor = self.getTravChipCoord(t_top_x, t_top_y, num)
        camera_offset_x = self.config["machine"]["distance_from_chip_to_cam"]["x"]
        camera_offset_y = self.config["machine"]["distance_from_chip_to_cam"]["y"]
        tgo.MoveLinear(300, (x_coor + camera_offset_x) * self.X_FAC, (y_coor + camera_offset_y) * self.Y_FAC,
                       self.config["machine"]["work_area_offset"]["z"] * self.Z_FAC, None)

    # move method
    # given clamshell top-left chip x,y move to a given chip num
    def move_cam_to_chip_in_clam(self, tgo, c_top_x, c_top_y, num):
        x_coor, y_coor = self.getClamChipCoord(c_top_x, c_top_y, num)
        camera_offset_x = self.config["machine"]["distance_from_chip_to_cam"]["x"]
        camera_offset_y = self.config["machine"]["distance_from_chip_to_cam"]["y"]
        tgo.MoveLinear(300, (x_coor + camera_offset_x) * self.X_FAC, (y_coor + camera_offset_y) * self.Y_FAC,
                       self.config["machine"]["work_area_offset"]["z"] * self.Z_FAC, None)

    # math method
    # get coordinates of a chip from top left chip within the traveler in mm
    def getTravChipCoord(self, t_top_x, t_top_y, num):
        chip_distance_x = self.config["physical"]["traveler"]["inner"]["x_inter"]
        chip_distance_y = self.config["physical"]["traveler"]["inner"]["y_inter"]
        chip_top_left_x = t_top_x + (int(num / 8) * chip_distance_x)
        chip_top_left_y = t_top_y - (int(num % 8) * chip_distance_y)

        chip_width_x = self.config["physical"]["traveler"]["inner"]["x_within"]
        chip_width_y = self.config["physical"]["traveler"]["inner"]["y_within"]
        return (chip_top_left_x + (chip_width_x / 2)), (chip_top_left_y - (chip_width_y / 2))

    # math method
    # get coordinates of a chip from top left chip within the clamshell in mm
    def getClamChipCoord(self, c_top_x, c_top_y, num):
        chip_distance_x = self.config["physical"]["clam_shell"]["inner"]["x_inter"]
        chip_distance_y = self.config["physical"]["clam_shell"]["inner"]["y_inter"]
        chip_top_left_x = c_top_x + (int(num / 2) * chip_distance_x)
        chip_top_left_y = c_top_y - (int(num % 2) * chip_distance_y)
        # return chip_top_left_x*self.X_FAC, chip_top_left_y*self.Y_FAC # top left of a chip
        chip_width_x = self.config["physical"]["clam_shell"]["inner"]["x_within"]
        chip_width_y = self.config["physical"]["clam_shell"]["inner"]["y_within"]
        return (chip_top_left_x + (chip_width_x / 2)), (chip_top_left_y - (chip_width_y / 2))

    # move method
    # move around traveler and then stich images
    def map_traveler(self, tgo, cam_short, x, y):  # takes top left start of traveler
        list_image = []
        chip_d_x = self.config["physical"]["chip"]["x"] + self.config["physical"]["traveler"]["inner"]["x_inter"]
        chip_d_y = self.config["physical"]["chip"]["y"] + self.config["physical"]["traveler"]["inner"]["y_inter"]

        for i in range(4):
            list_row = []
            for j in range(3):
                x_loc = (x + (chip_d_x * 3 * j)) * self.X_FAC
                y_loc = (y - (chip_d_y * 2 * i)) * self.Y_FAC
                tgo.MoveRapid(x_loc, y_loc, 0, 0)  # get a pictue with 2 rows of 4 chips
                _, img = cam_short.read()
                time.sleep(1)
                if self.DEBUG:
                    print("Going : ", x_loc, y_loc)
                    # cv2.imwrite(str(root / "dev" / "machine_learning" / "numberRecognition" / "chip_img_movement" /
                    #                 "output_imgs" / f"chipsC{i}xR{j}.jpg"), img)
                    cv2.imshow("cap", img)
                    cv2.waitKey(500)
                    list_row.append(img)
            if self.DEBUG:
                list_image.append(list_row)
        if self.DEBUG:
            img = wrp.stitch_image(list_image)
            cv2.imshow("jash", cv2.resize(img, (500, 500)))
            cv2.waitKey(1000)
        return img, list_image


# image method
# takes image and then thresholds it
def is_edge_in_center(img, darkness_thresh=235, thresh_val=150, white_thresh=210,
                      DEBUG=False):  # on num pad 5 is the traveler return 0 if not on edge return 5 if all white
    # pre process
    img_green_to_black = wrp.green_to_black(img, [150, 255, 150], [30, 30, 0])
    img_grey = cv2.cvtColor(img_green_to_black, cv2.COLOR_BGR2GRAY)
    cont, out1 = wrp.increaseContrast(cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB))
    ret, thresh_img = cv2.threshold(cv2.cvtColor(cont, cv2.COLOR_BGR2GRAY), thresh_val, white_thresh, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.erode(cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB), kernel, iterations=1)
    b_not = cv2.bitwise_not(cv2.cvtColor(dilated, cv2.COLOR_RGB2GRAY))

    h, w = np.shape(b_not)
    q1_img = b_not[:int(h / 2), int(w / 2):]
    q2_img = b_not[:int(h / 2), :int(w / 2)]
    q3_img = b_not[int(h / 2):, :int(w / 2)]
    q4_img = b_not[int(h / 2):, int(w / 2):]
    q1_mean = np.mean(q1_img)
    q2_mean = np.mean(q2_img)
    q3_mean = np.mean(q3_img)
    q4_mean = np.mean(q4_img)

    if DEBUG:
        out = [['original', img],
               ["Gray", cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB)],
               ["Contrast", cont],
               ["Thresh", cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)],
               ["erode", dilated],
               [f"q1:{int(q1_mean)}", cv2.cvtColor(q1_img, cv2.COLOR_GRAY2RGB)],
               [f"q2:{int(q2_mean)}", cv2.cvtColor(q2_img, cv2.COLOR_GRAY2RGB)],
               [f"q3:{int(q3_mean)}", cv2.cvtColor(q3_img, cv2.COLOR_GRAY2RGB)],
               [f"q4:{int(q4_mean)}", cv2.cvtColor(q4_img, cv2.COLOR_GRAY2RGB)],
               ["inverted", cv2.cvtColor(b_not, cv2.COLOR_GRAY2RGB)],
               ["map_green_to_white", img_green_to_black],
               ] + out1
        img = wrp.return_open_cv_img_from_list(out)
        cv2.imshow("Improcess output", img)
        cv2.waitKey(200)
    q1 = q1_mean > darkness_thresh
    q2 = q2_mean > darkness_thresh
    q3 = q3_mean > darkness_thresh
    q4 = q4_mean > darkness_thresh

    if DEBUG:
        print(q1, q2, q3, q4)
    return wrp.formate_edge_guess(q1, q2, q3, q4)

# image method
# takes image and then finds how far up the edge is in the image
def get_percent_of_img(edge_img, x=True, window_percent=0.4, search_perc=0.005, darkness_thresh=225, thresh_val=150,
                       white_thresh=210, DEBUG=False):
    # preprocess
    img_green_to_black = wrp.green_to_black(edge_img, [150, 255, 150], [30, 30, 0])
    img_grey = cv2.cvtColor(img_green_to_black, cv2.COLOR_BGR2GRAY)
    cont, out1 = wrp.increaseContrast(cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB))
    ret, thresh_img = cv2.threshold(cv2.cvtColor(cont, cv2.COLOR_BGR2GRAY), thresh_val, white_thresh, cv2.THRESH_BINARY)
    edges = wrp.edge_detection(thresh_img)

    # do edge detection for threshold
    im_height, im_width = edges.shape[:2]
    edge_x = int(wrp.edge_window_x(edges, int(im_width * window_percent), int(im_width * search_perc)))
    edge_y = int(wrp.edge_window_y(edges, int(im_height * window_percent), int(im_height * search_perc)))

    # Find percentage
    if x:
        if DEBUG:
            edge_color = edges.copy()
            edge_color = cv2.cvtColor(edge_color, cv2.COLOR_GRAY2RGB)
            cv2.line(edge_color, (edge_x, 0), (edge_x, im_height), (0, 0, 255), 2)
            imS = cv2.resize(edge_color, (960, 540))
            cv2.imshow("color x", imS)
            cv2.waitKey(1000)
        return edge_x / float(im_width)
    else:
        if DEBUG:
            edge_color = edges.copy()
            edge_color = cv2.cvtColor(edge_color, cv2.COLOR_GRAY2RGB)
            cv2.line(edge_color, (0, edge_y), (im_width, edge_y), (0, 0, 255), 2)
            imS = cv2.resize(edge_color, (960, 540))
            cv2.imshow("color y", imS)
            cv2.waitKey(1000)
        return edge_y / float(im_height)
