import sys
import traceback
import serial
from pathlib import Path
# custom libraries
import control_functions as cf
import get_chip_img
from helperscripts import findPaths as P
from tinyg import tinyg
import loop_functions as lf


# import chip finding object detection
# pathToChipImg  = p.make_path(["dev","machine_learning","chip_img_movement"])
# sys.path.insert(1, pathToChipImg)


if __name__ == '__main__':
    
    ## DISABLE USER INTERFACE
    headless_mode = True

    ## Create camera, pipe, and tinyg objects
    tgo = tinyg.tinyg_obj()
    move_obj = get_chip_img.chip_imgr()
    pipe = None
    cam_wide = None
    cam_short = None

    try:
        pipe, csvLstDict, cam_wide, cam_short, usingTweezer = lf.InitSteps(tgo, move_obj, headless_mode)
        lf.ControlLoop(tgo, move_obj, pipe, csvLstDict, cam_wide, cam_short, headless_mode, usingTweezer)
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
