import cv2
import os
import sys
import traceback

# Error States
class TooFewClamshells(Exception):
    pass

class UnrecoverableChipError(Exception):
    pass


# Removes tinyg.log since it appends
def remove_log(headless):
    if headless:
        print("Removing Logs")
    try:
        os.remove("tinyg.log")
    except:
        if headless:
            print("no logs found")
        pass


# Picks up chip from traveler slot and then drops it off
def PickUpChip(tgo, move_obj, trav_slot, trav_x, trav_y, usingTweezer):
    # Move over
    chipX, chipY = move_obj.getTravChipCoord(trav_x, trav_y, trav_slot)
    xdist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["x"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["x"]
    ydist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["y"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["y"]
    t_slot_x = (chipX + xdist) * move_obj.X_FAC
    t_slot_y = (chipY + ydist) * move_obj.Y_FAC
    tgo.MoveRapid(t_slot_x, t_slot_y, None, None)

    # Move down
    zdist = move_obj.config["machine"]["distance_to_actuator"]["tweezer"]["z"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_to_actuator"]["vacuum"]["z"]
    tgo.MoveRapid(None, None,  zdist * move_obj.Z_FAC, None)

    if(usingTweezer):
        # Put other side under tension
        tgo.MoveLinear(10, t_slot_x + (1 * move_obj.X_FAC), None, None, None)

        # Trigger actuator
        tgo.SolenoidOn()

    # Move up
    tgo.MoveRapid(None, None, move_obj.config["machine"]["work_area_offset"]["z"] * move_obj.Z_FAC, None)


# Picks up chip from traveler slot and then drops it off
def DropOffChip(tgo, move_obj, clam_slot, clam_x, clam_y, usingTweezer):
    # Find the top left chip in the clamshell
    chipX, chipY = move_obj.clamshell_chip_start(clam_x,clam_y)

    # Move over to clamshell slot
    chipX, chipY = move_obj.getClamChipCoord(chipX, chipY, clam_slot)
    xdist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["x"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["x"]
    ydist = move_obj.config["machine"]["distance_from_chip_to_actuator"]["tweezer"]["y"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_from_chip_to_actuator"]["vacuum"]["y"]
    c_slot_x = (chipX + xdist) * move_obj.X_FAC
    c_slot_y = (chipY + ydist) * move_obj.Y_FAC
    tgo.MoveLinear(300, c_slot_x, c_slot_y, None, None)

    # Move down
    zdist = move_obj.config["machine"]["distance_to_clamshell"]["tweezer"]["z"] if (usingTweezer) \
       else move_obj.config["machine"]["distance_to_clamshell"]["vacuum"]["z"]
    tgo.MoveLinear(300, None, None, zdist * move_obj.Z_FAC, None)
    
    # Drop it like its hot
    if(usingTweezer):
        tgo.SolenoidOff()
    
    # Move up
    tgo.MoveRapid(None, None, move_obj.config["machine"]["work_area_offset"]["z"]*move_obj.Z_FAC, None)


# Excepts if a chip to the right of the current one failed
def CheckUnrecoverableError(wrongChips, currTravSlotLoc):
    # Find chip number to the right of the current one
    errorNum = currTravSlotLoc + 8
    if errorNum > 63:
        return

    # Check equality with a chip in wrongChips
    if len(wrongChips) == 0:
        return
    for wc in wrongChips:
        if wc['Position on Chip Traveler'] == errorNum:
            if wc['Error Message'] != "ExistenceException":
                # Except with error message
                raise UnrecoverableChipError(wc['Error Message'] + " for chip ID {}".format(wc['Chip ID']))


# Adds to the list of wrong chips
def AddToWrongList(wrongChips, chip, message):
    print('CHIP ERROR with Message' + message)
    chip['Error Message'] = message
    wrongChips.append(chip)


# Write errors and messages to a log file
def WriteErrors(wrongChips, logPath="./chip_error.log"):
    with open(logPath, 'w') as f:
        for wc in wrongChips:
            f.write("Chip ID {} at traveler location {} could not be processed due to error: {}\n".format(
                wc['Chip ID'], wc['Position on Chip Traveler'], wc['Error Message']))


# Moves to a position out of the way of the overhead camera, without knocking into things
def MoveOutOfTheWay(tgo, move_obj):
    startbox_y = move_obj.config["machine"]["work_area_offset"]["y"]
    top_traveler_box_y = startbox_y + move_obj.config["physical"]["traveler_box"]["y"]
    tgo.MoveRapid(None, None, 0, None)
    tgo.MoveRapid(0, (startbox_y + top_traveler_box_y) * move_obj.Y_FAC, None, None)


# Calculates likely row for clamshell and gives precise x location
def ApproxWhichClamshellRow(approxLoc, move_obj):
    clam_r1 = move_obj.config["physical"]["constraints"]["clamshell_r1_center"]
    clam_r2 = move_obj.config["physical"]["constraints"]["clamshell_r2_center"]
    clam_r3 = move_obj.config["physical"]["constraints"]["clamshell_r3_center"]
    clam_r4 = move_obj.config["physical"]["constraints"]["clamshell_r4_center"]
    clam_diff_r1 = abs(approxLoc - clam_r1)
    clam_diff_r2 = abs(approxLoc - clam_r2)
    clam_diff_r3 = abs(approxLoc - clam_r3)
    clam_diff_r4 = abs(approxLoc - clam_r4)
    diff_list = [clam_diff_r1, clam_diff_r2, clam_diff_r3, clam_diff_r4]
    min_index = diff_list.index(min(diff_list)) + 1
    print(f"Found row {min_index}")
    return min_index, move_obj.config["physical"]["constraints"]["clamshell_r" + str(min_index) + "_top"]

# Courtesy of https://stackoverflow.com/questions/4690600/python-exception-message-capturing
def PrintDetailedException():
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
    stack_trace = list()
    for trace in trace_back:
        stack_trace.append(
            "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" % ex_value)
    print("Stack trace : %s" % stack_trace)