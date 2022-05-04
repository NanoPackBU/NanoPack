import os
import time
import sys
import traceback
from pathlib import Path
import glob

ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in range(100):
        if (len(glob.glob(str(new_path / ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path
#
root = get_root()

import tinyg_spoofer as tinyg

def from_dbids_test_file_test():
     print("here")
     ## Delete Log File if exists
     try:
       os.remove("tinyg.log")
     except:
       pass
     try:
       ## Setup serial connection
       tgo = tinyg.tinyg_obj()
       tgo.SetupConnection()
       print("Port of serial connection: {}".format(tgo.ser.port))
       if tgo.ser == 0:
         ##Tell the GUI we need manual port
         pass

       ## Check config settings
       tgo.Config(tinyg_path)

       ## Home
       tgo.Home()

       ## Do some test movement
       #MAX_X = 20
       #MAX_Y = 11.8
       #MAX_Z = 83.34

       #tgo.SavePos1()
       tgo.MoveRapid(80, 100, None, None)
       #tgo.GoPos1()
       #tgo.MoveLinear(100, 10, None, None, None)
       #tgo.GoPos1()
       # tgo.SolenoidOn()
       # time.sleep(1)
       # tgo.SolenoidOff()

       tgo.CloseConnection()

     except tinyg.TinygThreadException as ex:
       print("TINYG Exception")
       ## Communicate to the GUI that there is an error on the board

       ## Close serial link and runaway read thread
       tgo.CloseConnection()

     except Exception:
       print("\n...................................EXCEPTION...................................")
       ex_type, ex_value, ex_traceback = sys.exc_info()
       trace_back = traceback.extract_tb(ex_traceback)
       stack_trace = list()
       for trace in trace_back:
           stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
       print("Exception type : %s " % ex_type.__name__)
       print("Exception message : %s" %ex_value)
       print("Stack trace : %s" %stack_trace)

       ## Close serial link and runaway read thread
       if (tgo.ser is not None):
         tgo.CloseConnection()
      return True

if __name__ == "__main__":
    result = from_dbids_test_file_test()
