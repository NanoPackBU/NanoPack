import cv2
import numpy as np

def extract_slots(clamshell):
    img = cv2.imread(clamshell)
    gray = cv2.cvtColor(img, cv2.COLOR_BRG_GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv2.imshow("threshed", thresh)




clamshell = "C:\\NanoView_G33\\dev\\machine_learning\\numberRecognition\\clamshell_photos\\slot1.jpg"
extract_slots(clamshell)
