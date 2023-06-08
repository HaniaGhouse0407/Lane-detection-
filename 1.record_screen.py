import time

import cv2
import numpy as np  # Numerical Python. It is a Python library used for working with an array. In Python, we use the list for purpose of the array but it’s slow to process. NumPy array is a powerful N-dimensional array object and its use in linear algebra, Fourier transform, and random number capabilities. It provides an array object much faster than traditional Python lists.
import pyautogui
from PIL import \
    ImageGrab  # python imaging library which provides python with image editing capabilities


import cv2
import os



last_time=time.time()#takes the current time
while(True):
    printscreen=np.array(ImageGrab.grab(bbox=(0,40,1000,1000)))#takes a screenshot of the screen 
    print('loop {} seconds',format(time.time()-last_time)) #formats the time while running the loop - the prev time
    last_time=time.time()#updates time
    cv2.imshow("window",printscreen)
    if cv2.waitKey() & 0xFF== ord("q"): 
        cv2.destroyAllWindows
        break