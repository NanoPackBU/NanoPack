import cv2
import numpy as np
import random as rd

class Fake_cam():
    def __init__(self,id,img=[]):
        self.id = id
        if len(img) ==0:
            self.img = canvas = np.zeros((100,100,3))+3
            return
        self.img = img
    def read(self):
        success = rd.randint(0,100)
        return success>5,self.img
