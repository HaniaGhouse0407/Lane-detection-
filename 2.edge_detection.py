import time

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab

#from directkeys import A, D, PressKey, S, W


def process_img(image):
    original_image=image
    processed_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image=cv2.Canny(processed_image,threshold1=200,threshold2=300)
    return processed_image



last_time=time.time()
while(True):
         #PressKey(W)
    screen= np.array(ImageGrab.grab(bbox=(0,40,1000,1000)))
    new_screen=process_img(screen)
    last_time=time.time()
    cv2.imshow("window",new_screen)
    if cv2.waitKey() & 0xFF==ord("q"):
        cv2.destroyAllWindows
        break