import control_functions as cf
import sys
import cv2
# custom libraries
from helperscripts import csv_parse
from helperscripts import findPaths as P
from machine_learning import digit_detection_pl as numpie

isWindows = sys.platform.startswith('win')
if isWindows:
    from helperscripts import Pipe
    pathChar = "\\"
else:
    pathChar = "/"


def InitSteps(tgo, move_obj, headless):
    print("\n...................................Initial Setup...................................")

    # Start the warmup camera 'thread'
    # global cam_th
    # cam_th = tinyg.TinygThread(target=start_cam, args=[move_obj])

    # cam_th.start()
    wide, short = move_obj.start_cam()
    # ret,_ = move_obj.cam_short.read()
    cf.remove_log(headless)

    # .NET Pipe
    pipe = None
    csv_path = P.make_path(["src"]) + pathChar + "Quicktest.csv"
    usingTweezer = False
    if isWindows and not headless:
        pipe = Pipe.PipeServer("Test Pype1")
        pipe.connect()
        print("Pipe connected, waiting for CSV")
        
        # Get CSV path
        pipe.write("CSVRequested")
        buf = pipe.read()
        csv_path = buf[0]
        print("Path from .NET: ", csv_path)

        # Get whether tweezer or vacuum is selected
        # pipe.write("TweezerRequested")
        # buf = pipe.read()
        # usingTweezer = buf[0]
        # print("Using Tweezer" if (usingTweezer) else "Using Vacuum")

    # Setup serial connection automatically with TinyG
    try:
        tgo.SetupConnection()
        # tell pipe the chosen serial port
        if isWindows and not headless:
            pipe.write("SerialPort:{}".format(tgo.ser.port))
    except:
        # ask pipe for serial port manually and block until it replies
        if isWindows and not headless:
            pipe.write("SerialPort:NotFound")
            buf = pipe.read()
            tgo.SetupConnection(buf)
        else:
            print("Cannot automatically find the serial connection")
            sys.exit("Serial Failure")
    print("Found Serial Port: {}".format(tgo.ser.port))

    # Parse CSV to list of dictionaries
    csvLstDict = csv_parse.parse_csv(csv_path)

    # Order List Correctly
    csvLstDict = csv_parse.find_chip_order(csvLstDict)
    # Configure the TinyG
    tgo.Config(P.make_path(["src", "tinyg"]))

    # Home the TinyG
    tgo.Home()

    # Move to the beginning of the box
    cf.MoveOutOfTheWay(tgo, move_obj)

    # Wait for the warmup camera 'thread' to complete
    # cam_th.join()

    return pipe, csvLstDict, wide, short, usingTweezer


def ControlLoop(tgo, move_obj, pipe, csvLstDict, cam_wide, cam_short, headless, usingTweezer):
    # Find General Locations of Traveler and Clamshells (YOLO)
    x_of_traveler_in_bed, y_of_traveler_in_bed, clamshellLocs = move_obj.find_general_loc(cam_wide, attempts=3)
    
    # TEMP FIXED CLAMSHELL LOCS
    # clamshellLocs = []
    # for i in range(4):
    #     temp = {}
    #     temp['Class'] = 'Clamshell'
    #     temp['X_center'] = 275
    #     temp['Y_center'] = move_obj.config["physical"]["constraints"]["clamshell_r" + str(i+1) + "_center"]
    #     clamshellLocs.append(temp)
    
    print("DONE ML :", x_of_traveler_in_bed, y_of_traveler_in_bed, clamshellLocs, "\n\n\n\n")

    # Find Traveler Precise Location (Edge Detection)
    trav_top_x, trav_top_y = move_obj.find_top_traveler(tgo, cam_short, (x_of_traveler_in_bed, y_of_traveler_in_bed))
    
    # TEMP FIXED TRAVELER
    # trav_top_x = move_obj.config["physical"]["constraints"]["traveler"]["x"]
    # trav_top_y = 256.75
    # clamshellLocs = [{"X_center" : 100, "Y_center" : 100 }]

    topx, topy = move_obj.traveller_chip_start(trav_top_x, trav_top_y)

    # TEMP: TEST TOP LEFT
    # tgo.MoveLinear(300, topx*move_obj.X_FAC, topy*move_obj.Y_FAC, None, None)
    # img = move_obj.readImg(cam_short, 2, reticle=True)
    # cv2.imshow("cap top left", img)
    # cv2.waitKey(0)

    # Data Structures for Loop
    wrongChips = []
    numClamshellFound = len(clamshellLocs)
    packedClams = []
    for _ in range(numClamshellFound):
        # first four = id of loaded chip, last two = location of clamshell
        packedClam = [-1, -1, -1, -1, -1, -1]
        packedClams.append(packedClam)

    # CHIP LOOP
    print(csvLstDict)
    for idx, chip in enumerate(csvLstDict):
        print(f"\n...................................LOOP {idx}...................................")
        
        # Verify that no error occurred to the right of this chip for the tweezer actuator
        if(usingTweezer):
            try:
                cf.CheckUnrecoverableError(wrongChips, chip['Position on Chip Traveler'])
            except cf.UnrecoverableChipError as e:
                print("Unrecoverable Chip Error with message: {}".format(e.args[0]))
                if isWindows and not headless:
                    pipe.write("Unrecoverable Chip Error with message: {}".format(e.args[0]))
                    buf = pipe.read()
                    if str(buf) == "Exit":
                        raise
                else:
                    cf.WriteErrors(wrongChips, "./chip_error.log")
                    raise

        # Move to location of Chip in Traveler
        move_obj.move_cam_to_chip_in_trav(tgo, topx, topy, chip['Position on Chip Traveler'])
        img = move_obj.readImg(cam_short, 2, reticle=True)
        cv2.imshow("cap chip", img)
        cv2.waitKey(1000)

        # Verify a chip exists in that location
        chipExists = True

        # If chip doesn't exist save to list of wrong chips
        if not chipExists:
            cf.AddToWrongList(wrongChips, chip, "ExistenceException")
            continue

        # Verify Number of Chip in that location
        img = move_obj.readImg(cam_short, 2, reticle=True)
        cv2.putText(img,'Chip Guess ', (int(img.shape[0]*0.2),int(img.shape[1]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite("output.png", img)
        
        try:
            guess = numpie.trav_img(img)
            print("chip id:", chip['Chip ID'], "guess:", guess)
            chipCorrect = int(chip['Chip ID']) == int(guess)
            print("Chip Num Correct" if(chipCorrect) else "Chip Num Incorrect")
            img = move_obj.readImg(cam_short, 2, reticle=True)
            cv2.putText(img,f'Chip Guess {str(guess)}', (int(img.shape[0]*0.2),int(img.shape[1]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 1, cv2.LINE_AA)
            cv2.imwrite("output.png", img)
        except:
            ex_type, ex_value, ex_traceback = sys.exc_info()
            print("NUMBER RECOGNTION ERROR")
            print("Exception type : %s " % ex_type.__name__)
            print("Exception message : %s" % ex_value)
            chipCorrect=False

        # Number not recognized
        if not chipCorrect:
            # Prompt GUI for whether to fix
            if isWindows and not headless:
                pipe.write("NumberNotRecognized: Chip ID {}, Traveler Loc {}".format(chip['Chip ID'],
                                                                                     chip['Position on Chip Traveler']))
                buf = pipe.read()
                if str(buf) == "DontIgnore":
                    cf.AddToWrongList(wrongChips, chip, "Chip ID not recognized by ML")
            else:
                cf.AddToWrongList(wrongChips, chip, "Chip ID not recognized by ML")

        # Determine that the clamshell is present
        if (numClamshellFound - 1) <= chip['Chip Package Number']:
            cf.AddToWrongList(wrongChips, chip, "Too few clamshells found to load")
            continue

        # Determine that the clamshell slot is open
        if (packedClams[chip['Chip Package Number']][chip['Position In Chip Package']]) != -1:
            cf.AddToWrongList(wrongChips, chip, "Clamshell slot already full")
            continue

        # Pick up the Chip
        cf.PickUpChip(tgo, move_obj, chip['Position on Chip Traveler'], topx, topy, usingTweezer)

        # Check Clamshell Location Known
        if packedClams[chip['Chip Package Number']][4] == -1 or packedClams[chip['Chip Package Number']][5] == -1:
            # Find exact position of top-left of clamshell with edge detection
            row_num, clam_top_y = cf.ApproxWhichClamshellRow(clamshellLocs[chip['Chip Package Number']]["Y_center"], move_obj)
            clam_top_x, _ = move_obj.find_top_clamshell(tgo, cam_short, (clamshellLocs[chip['Chip Package Number']]["X_center"],
                                    move_obj.config["physical"]["constraints"]["clamshell_r" + str(row_num) + "_center"]),
                                    constrain_row=row_num)
            packedClams[chip['Chip Package Number']][4] = clam_top_x
            packedClams[chip['Chip Package Number']][5] = clam_top_y

        # Drop off the Chip
        # print("############ TEST ############")
        # print(f"chip['Position In Chip Package'] {chip['Position In Chip Package']}, packedClams[chip['Chip Package Number']][4] {packedClams[chip['Chip Package Number']][4]}, \
        #     packedClams[chip['Chip Package Number']][5] {packedClams[chip['Chip Package Number']][5]}, usingTweezer{usingTweezer}")
        # tgo.MoveLinear(300, clam_top_x * move_obj.X_FAC, clam_top_y * move_obj.Y_FAC, None, None)
        # img = move_obj.readImg(cam_short, 2, reticle=True)
        # cv2.imshow("cap top left", img)
        # cv2.waitKey(0)
        # return
        cf.DropOffChip(tgo, move_obj, chip['Position In Chip Package'], packedClams[chip['Chip Package Number']][4], \
            packedClams[chip['Chip Package Number']][5], usingTweezer)

        # Record Chip as Moved
        packedClams[chip['Chip Package Number']][chip['Position In Chip Package']] = chip['Chip ID']

    # If too few clamshells found, error and report to GUI
    maxCSVClamNum = max(csvLstDict, key=lambda x: x['Chip Package Number'])['Chip Package Number']
    if (numClamshellFound - 1) < maxCSVClamNum:
        raise cf.TooFewClamshells

    # If complete, report chip locations and errors to the GUI
    if isWindows and not headless:
        pipe.write("Packing Complete with {} placed chips and {} errors")

    # Write errors and messages to a log file
    cf.WriteErrors(wrongChips, "./chip_error.log")


def CloseSteps(tgo, headless, pipe=None, cam_wide=None, cam_short=None, errorCode=None):
    print("Running close steps")

    # Close Pipe
    if isWindows and not headless and pipe is not None:
        print("closing pipe")
        pipe.write("Exited with code: " + errorCode)

        # close pipe
        pipe.write(None)
        pipe.close()

    # Close serial link and runaway read thread
    if tgo.ser is not None:
        print("closing serial link")
        tgo.CloseConnection()

    # Close camera connection
    print("closing camera")
    cv2.destroyAllWindows()
    if cam_short is not None and cam_short.isOpened():
        cam_short.release()
    if cam_wide is not None and cam_wide.isOpened():
        cam_wide.release()
