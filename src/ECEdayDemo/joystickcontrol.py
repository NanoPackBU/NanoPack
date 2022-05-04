import pygame
import time
import glob
from pathlib import Path
import sys
ROOT = ".git"
def get_root():
    new_path = Path(__file__)
    for i in range(100):
        if (len(glob.glob(str(new_path / ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path

root = get_root()
pathToTinyg = Path(glob.glob(str(root)+"/**/*tinyg.py",recursive = True)[0]).parent.absolute()
sys.path.insert(1, str(pathToTinyg))
import tinyg 

pygame.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)
controller.init()
X = 0
Y = 0
Z = 0
TRIGR = 5 
RIGHTY = 3
RIGHTX = 2

TRIGL = 4
LEFTY = 1
LEFTX =0

X_BUTT = 0
O_BUTT = 1
TRI_BUTT = 2
BOX_BUTT = 3
tgo = tinyg.tinyg_obj()
tgo.SetupConnection()
tgo.Home()

Move_X = 0.5
Move_Y = 0.5 
Move_Z = 0.5
SPEED = 0.5
X = 0 
Y = 0 
Z = 0 
Tresh = 0.1
close = False 
while True:
            dX = 0 
            dy = 0 
            dz = 0 
            for event in pygame.event.get():
                            if event.type == pygame.JOYAXISMOTION:
                                
                                dic = event.dict
                                # if(dic["axis"] == 3): print("joystick,",dic)
                                if (dic["axis"] == LEFTY):
                                    dy = -1*Move_Y*event.value*(abs(event.value)>Tresh)
                                if (dic["axis"] == LEFTX):
                                    dX = Move_X*event.value*(abs(event.value)>Tresh)
                                if (dic["axis"] == RIGHTY):
                                    dz = Move_Z*event.value*(abs(event.value)>Tresh)
                            elif event.type == pygame.JOYBALLMOTION:
                                print("JOYBALLMOTION") #event.dict, event.joy, event.ball, event.rel)
                            # elif event.type == pygame.JOYBUTTONDOWN:
                            #     print(event.dict, event.joy, event.button, 'pressed')
                            elif event.type == pygame.JOYBUTTONUP:
                                print(event.dict, event.joy, event.button, 'released')
                                if (event.button == X_BUTT):
                                    close = not close 
                                    if(close):
                                        tgo.SolenoidOff()
                                    else:
                                        tgo.SolenoidOn()
                                    

                            # elif event.type == pygame.JOYHATMOTION:
                            #     print(event.dict, event.joy, event.hat, event.value)
            # print(dX,dy,dz)
            # dX = 0 if dX<0.01 else dX
            # dy = 0 if dy<0.01 else dy 
            # dz = 0 if dz<0.01 else dz 
            
            if(abs(dX) > 0 or abs(dy) >0 or abs(dz)> 0 ): 
               
                X += dX
                Y += dy 
                Z += dz 
                X = 0 if X<0.001 else X
                Y = 0 if Y<0.001 else Y 
                Z = 0 if Z<0.001 else Z 
                # print("Jogging: ", X, Y, Z)
                tgo.MoveLinear(200, X, Y, Z, None)
                # print(close)
            time.sleep(0.01)
