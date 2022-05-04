headless_mode = True
####----------BUILT-IN-LIBS----------####
from shutil import move
import sys
import traceback
import serial
from pathlib import Path
import cv2
import os
import glob

####----------CUSTOM-LIBS----------####
# get helper scripts
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
#
root = get_root()
#
helperDir = Path(glob.glob(str(root)+"/**/*findPaths.py",recursive = True)[0]).parent.absolute()
# sys.path.insert(1, str(path_to_yolo))

# startingdir = os.getcwd()+pathChar # gets starting directort
# path = Path(startingdir)
# #srcdir = Path(path.parent.absolute()) # assumes that it is exactly 2 layers down
# NanoView_G33 = Path(path.parent.absolute()) # hopefully this is nanoviewg33
# helperDir = str(NanoView_G33)+pathChar+"dev"+pathChar+"helperscripts"+pathChar# get the longform of the helper directory
sys.path.insert(1, str(helperDir))
import findPaths as p

# import chip finding object detection
pathToSrc = Path(glob.glob(str(root)+"/**/*config_distance.yml",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(pathToSrc))

import control_functions as cf
import get_chip_img
from helperscripts import findPaths as P
from tinyg import tinyg
import loop_functions as lf
from helperscripts import csv_parse

# Picks up chip from traveler slot and then drops it off
def PickUpChip(tgo, move_obj, trav_slot, trav_x, trav_y, usingTweezer):
    # Move over
    chipX, chipY = move_obj.getTravChipCoord(trav_x, trav_y, trav_slot)
    xdist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["x"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["x"]
    ydist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["y"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["y"]
    t_slot_x = chipX + xdist * move_obj.X_FAC
    t_slot_y = chipY + ydist * move_obj.Y_FAC
    tgo.MoveRapid(t_slot_x, t_slot_y, None, None)

    # Move down
    zdist = move_obj.config["machine"]["distance_to_actuator"]["tweezer"]["z"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_to_actuator"]["vacuum"]["z"]
    tgo.MoveRapid(None, None,  * move_obj.Z_FAC, None)

    # Put other side under tension
    tgo.MoveLinear(10, t_slot_x + (1 * move_obj.X_FAC), None, None, None)

    # Trigger actuator
    if(usingTweezer):
        tgo.SolenoidOn()

    # Move up
    tgo.MoveRapid(None, None, move_obj.config["machine"]["work_area_offset"]["z"] * move_obj.Z_FAC, None)


# Picks up chip from traveler slot and then drops it off
def DropOffChip(tgo, move_obj, clam_slot, clam_x, clam_y, usingTweezer):
    # Move over to clamshell slot
    chipX, chipY = move_obj.getClamChipCoord(clam_x, clam_y, clam_slot)
    xdist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["x"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["x"]
    ydist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["y"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["y"]
    c_slot_x = (chipX + xdist) * move_obj.X_FAC
    c_slot_y = (chipY + ydist) * move_obj.Y_FAC
    tgo.MoveLinear(300, c_slot_x, c_slot_y, None, None)

    img = move_obj.readImg(cam_short, 3, iters=10, reticle=True)
    cv2.imshow("cap act", img)
    cv2.waitKey(0)

    # Move down
    zdist = move_obj.config["machine"]["distance_to_actuator"]["tweezer"]["z"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_to_actuator"]["vacuum"]["z"]
    # tgo.MoveLinear(300, None, None, zdist * move_obj.Z_FAC, None)
    
    # Drop it like its hot
    if(usingTweezer):
        tgo.SolenoidOff()
    
    # Move up
    tgo.MoveRapid(None, None, move_obj.config["machine"]["work_area_offset"]["z"]*move_obj.Z_FAC, None)

####----------CONTROL-LOOP----------####
def ControlLoop(tgo, move_obj, pipe, csvLstDict, cam_wide, cam_short, usingTweezer):

    ## BEGIN TRAVELER TEST
    x_of_traveler_in_bed = move_obj.config["physical"]["constraints"]["traveler"]["x"]
    y_of_traveler_in_bed = 256.75
    
    # topx = x_of_traveler_in_bed
    # topy = y_of_traveler_in_bed

    # tgo.MoveLinear(300, x_of_traveler_in_bed*move_obj.X_FAC, y_of_traveler_in_bed*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 1, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    # topx,topy =  move_obj.traveller_chip_start(x_of_traveler_in_bed, y_of_traveler_in_bed)
    # tgo.MoveLinear(300, topx*move_obj.X_FAC, topy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 1, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    # # chipLocs = [10, 25]
    # # for num, chipL in enumerate(chipLocs):
    # #     move_obj.move_cam_to_chip_in_trav(tgo,topx,topy,chipL)
    # #     img = move_obj.readImg(cam_short, 2, reticle=False)
    # #     cv2.imwrite("../chip_photos/"+str(num + 2000)+".png", img)
    # #     if (num == 0):
    # #         cv2.imshow("cap 1", img)
    # #         cv2.waitKey(0)
    # move_obj.move_cam_to_chip_in_trav(tgo,topx,topy,10)
    # img = move_obj.readImg(cam_short, 1, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    # chipX, chipY = move_obj.getTravChipCoor(topx,topy,10)
    # tgo.MoveLinear(300, chipX, chipY, None, None)
    # img = move_obj.readImg(cam_short, 1, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    # PickUpChip(tgo, move_obj, 10, topx, topy)
    # img = move_obj.readImg(cam_short, 2, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    # clamx = move_obj.config["phyisical"]["constraints"]["clamshell_r1_left"]["x"]
    # clamy = 253.6
    # chipX, chipY = move_obj.clamshell_chip_start(clamx,clamy)
    # print("Clamshell top left",chipX, chipY)
    # DropOffChip(tgo, move_obj, 3, chipX, chipY)
    # img = move_obj.readImg(cam_short, 2, reticle=True)
    # cv2.imshow("cap 1", img)
    # cv2.waitKey(0)

    ## END TRAVELER TEST

    ## BEGIN CLAMSHELL ROW TEST
    
    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r1_top"]
    # clamx = 200
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r1t", img)
    # cv2.waitKey(0)
    
    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r1_center"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r1c", img)
    # cv2.waitKey(0)

    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r2_top"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r2t", img)
    # cv2.waitKey(0)
    
    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r2_center"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r2c", img)
    # cv2.waitKey(0)

    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r3_top"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r3t", img)
    # cv2.waitKey(0)
    
    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r3_center"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r3c", img)
    # cv2.waitKey(0)

    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r4_top"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r4", img)
    # cv2.waitKey(0)
    
    # clamy = move_obj.config["physical"]["constraints"]["clamshell_r4_center"]
    # tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 3, reticle=True)
    # cv2.imshow("cap r4", img)
    # cv2.waitKey(0)

    ## END CLAMSHELL ROW TEST

    ## BEGIN WITHIN CLAMSHELL TEST

    # set top left
    clamy_center = move_obj.config["physical"]["constraints"]["clamshell_r2_center"]
    clamy = move_obj.config["physical"]["constraints"]["clamshell_r2_top"]
    clamx = 200
    print(clamx, clamy)
    tgo.MoveLinear(300, clamx*move_obj.X_FAC, clamy_center*move_obj.Y_FAC, None, None)
    # while(True): ## use to config
    img = move_obj.readImg(cam_short, 3, iters=10, reticle=True)
    cv2.imshow("cap 2", img)
    cv2.waitKey(0)
    
    # get top left chip
    chipX, chipY = move_obj.clamshell_chip_start(clamx,clamy)
    print(chipX, chipY)
    tgo.MoveLinear(300, chipX*move_obj.X_FAC, chipY*move_obj.Y_FAC, None, None)
    img = move_obj.readImg(cam_short, 3, iters=10, reticle=True)
    cv2.imshow("cap 2", img)
    cv2.waitKey(0)

    # go through the middle of the slots
    for k in range(4):
        tempX, tempY = move_obj.getClamChipCoord(chipX,chipY,k)
        print(tempX, tempY)
        tgo.MoveLinear(300, tempX*move_obj.X_FAC, tempY*move_obj.Y_FAC, None, None)
        img = move_obj.readImg(cam_short, 3, iters=20, reticle=True)
        cv2.imshow("cap 2", img)
        cv2.waitKey(0)
    
    #DropOffChip(tgo, move_obj, 3, clamx, clamy, usingTweezer)

    ## END WITHIN CLAMSHELL TEST

    return

####----------MAIN----------####
if __name__ == '__main__':
    tgo = tinyg.tinyg_obj()
    move_obj = get_chip_img.chip_imgr()
    pipe = None
    cam_wide = None
    cam_short = None
    try:
        pipe, csvLstDict, cam_wide, cam_short, usingTweezer = lf.InitSteps(tgo, move_obj, headless_mode)
        ControlLoop(tgo, move_obj, pipe, csvLstDict, cam_wide, cam_short, usingTweezer)
        raise SystemExit

    except cf.TooFewClamshells:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'TooFewClamshells')
        except:
            pass
    except get_chip_img.TravelerNotFound:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'TravelerNotFound')
        except:
            pass
    except get_chip_img.ClamshellsNotFound:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'ClamshellsNotFound')
        except:
            pass
    except tinyg.TinygThreadException:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'TinygThreadException')
        except:
            pass
    except cf.UnrecoverableChipError:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'UnrecoverableChipError')
        except:
            pass
    except serial.serialutil.SerialException:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'SerialPortNotFound')
        except:
            pass
    except KeyboardInterrupt:
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'KeyboardInterrupt')
        except:
            pass
    except TimeoutError:
        print("TimeoutError")
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'TimeoutError')
        except:
            pass
    except AssertionError:
        print("CSV chip nums not equal to scanned nums")
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'CSVError')
        except:
            pass
    except SystemExit:
        print("Exited normally")
        try:
            # Move TinyG for Faster Startup
            cf.MoveOutOfTheWay(tgo, move_obj)

            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'NormalExit')
        except:
            pass
    except Exception as e:
        # Courtesy of https://stackoverflow.com/questions/4690600/python-exception-message-capturing
        print("\n...................................EXCEPTION...................................")
        ex_type, ex_value, ex_traceback = sys.exc_info()
        trace_back = traceback.extract_tb(ex_traceback)
        stack_trace = list()
        for trace in trace_back:
            stack_trace.append(
                "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
        print("Exception type : %s " % ex_type.__name__)
        print("Exception message : %s" % ex_value)
        print("Stack trace : %s" % stack_trace)
        try:
            lf.CloseSteps(tgo, headless_mode, pipe, cam_wide, cam_short, 'Other Error')
        except:
            pass
    sys.exit(1)

