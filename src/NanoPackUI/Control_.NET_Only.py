"""
- ####----------OUTLINE----------####
- import img pos processing
- import serial talk
- import gui communication
- import motor control logic

- create necessary objects

- init steps - after each bullet update the file for gui
    - connect to serial port
    - setup GUI pipe
    - create image mapping factor
        - find context lines

- scan chip for errors
  - read config file, define global chip number array
  - scan all chip numbers and relative locs into local vars
  - check chip locations against csv file locations
  - return error if not equal

- control loop
    - Get the location of chip
    - Get the location of the clamshell
    - Listen for GUI flags
    - If GUI for arduino send
      - Send difference of locations to be displayed on the arduino
    - update GUI on progress
"""
####----------BUILT-IN-LIBS----------####
import time
from enum import Enum
import random
import math
import sys
import traceback

isWindows = sys.platform.startswith('win')

import win32file
import win32pipe
# import pySerialFunction
import cv2 as cv

####----------CUSTOM-LIBS----------####
import CSVparse

if isWindows:
    import Pipe
# from ChipRecognizer import Camera
# from ChipRecognizer import Locationscan


####----------GLOBAL-VARS----------####
# chipNums = [0 for i in range(64)]

####----------INITIALIZE----------####
def InitSteps():
    ## Setup serial connection
    # link = pySerialFunction.SetupConnection()

    ## TODO Setup .NET Pipe
    ## assignees: jem2000
    pipe = None
    if isWindows:
        pipe = Pipe.PipeServer("Test Pype1")
        pipe.connect()
    ## create and pass a camera object
    # locationCam = Camera(0)
    locationCam = None
    locationScanner = None
    ## create and pass the object scanning function
    # locationScanner = Locationscan(".\\dev\\generateTestImagesMacro\\Images\\Train_1.png",
    #                                ".\\dev\\generateTestImagesMacro\\Outputs\\scanned.png")
    # return link, pipe, locationCam, locationScanner
    return pipe, locationCam, locationScanner


####----------SCAN CHIP NUMBERS----------####
def ScanChipNum(pipe, locScan):
    ## TODO Create the image mapping factor
    ## assignees: pequode
    # testNum =2
    ## Read config CSV
    csvDict = None
    if isWindows:
        print("\n...................................Reading CSV...................................")
        pipe.write("CSVRequested")
        buf = pipe.read()
        print("Path from .NET: ", buf[0])
        pathForim = str(buf[0])[str(buf[0]).find("Train_") + 6: str(buf[0]).find(".csv")]
        # testNum = int(pathForim)
        csvDict = CSVparse.parseCSV(buf[0])
        for key, value in csvDict.items():
            print(key, '->', value)

            ## TODO Scan chip numbers into local var (text recognition)
    ## assignees: pequode

    ## Scan chip locations into local var
    print("\n...................................Chip Locations from Sample Image...................................")
    # locScan.inPath = "C:\\Users\\dbids\\NanoView_G33\\\dev\\generateTestImagesMacro\\Images\\Train_{im}.png".format(
    #     im=testNum)
    # locScan.readWrite()
    # chipLocs = locScan.identifiedSquares
    # numChips = len(chipLocs)
    # print(chipLocs)

    ## TODO Compute relative location of chip nums (i.e. where is it in 16x16 grid)
    ## assignees: pequode

    ## TODO Compare CSV to scanned information use 'assert' keyword instead
    ## assignees: dbids
    CSVdiscrepancy = 0
    if CSVdiscrepancy:
        raise AssertionError
    numChips = 6
    chipLocs = []
    return csvDict, numChips, chipLocs


####----------CONTROL-LOOP----------####
def ControlLoop(link, pipe, count, chipLocs, csvDict):
    print("\n...................................LOOP {}...................................".format(count))
    # chipLocs = [[i+j for i in range(2)] for j in range(len(chipLocs))]
    ## TODO Get the location of chip
    ## assignees: pequode

    clamshellLoc = [[i * j for i in range(2)] for j in range(len(chipLocs))]
    ## TODO Get the location of the clamshell slots
    ## assignees: pequode

    ## TODO Listen to .NET pipe and set "startPacking" that indicates if we are starting
    ## assignees: jem2000
    startPacking = 1  # Needs to be changed to some non-blocking form of reading

    # if startPacking:
    #     ## Print the amount we need to move the first chip to the clamshell
    #     deltaX, deltaY = pySerialFunction.CalcMove(chipLocs[count][0], chipLocs[count][1],
    #                                                clamshellLoc[count][0], clamshellLoc[count][1])
    #     pySerialFunction.SendFunction(link, pySerialFunction.FunctionCode.MOVE_NUM.value, csvDict['ChipNumber'][count],
    #                                   deltaX, deltaY)

    pipe.write("Sending packing report " + str(count))
    buf = pipe.read()
    print("Message from .NET GUI: " + str(buf[0]) + "\n")
    time.sleep(1)


####----------CLOSE----------####
def CloseSteps(link, pipe, errorCode):
    ## TODO Send Error Code back to GUI to be displayed to user
    ## assignees: jem2000
    if (isWindows):
        pipe.write("Exited with code: " + errorCode)

        ## close pipe
        print("closing pipe")
        pipe.write(None)
        pipe.close()

    ## Close serial connection
    # link.close()


####----------MAIN----------####
if __name__ == '__main__':
    try:
        # link, pipe, locCam, locScan = InitSteps()
        pipe, locCam, locScan = InitSteps()
        link = None
        csvDict, numChips, chipLocs = ScanChipNum(pipe, locScan)
        numChips = 3
        count = 0
        while True:
            ControlLoop(link, pipe, count, chipLocs, csvDict)
            count += 1
            if count == numChips:
                raise SystemExit

    except KeyboardInterrupt:
        try:
            CloseSteps(link, pipe, 'KeyboardInterrupt')
        except:
            pass
    except TimeoutError:
        print("TimeoutError")
        try:
            CloseSteps(link, pipe, 'TimeoutError')
        except:
            pass
    except AssertionError:
        print("CSV chip nums not equal to scanned nums")
        try:
            CloseSteps(link, pipe, 'CSVError')
        except:
            pass
    except SystemExit:
        print("Exited normally")
        try:
            CloseSteps(link, pipe, 'NormalExit')
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
            CloseSteps(link, pipe, 'Other Error')
        except:
            pass
            sys.exit(1)
