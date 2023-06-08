import time

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab


def roi(img,vertices):
    mask= np.zeros_like(img)# creates an empty array of same shape and size as the img
    cv2.fillPoly(mask,vertices,255)# used to draw polyd=sons like rectangle etc. vertices- points/coordinates. 255- colour
    masked=cv2.bitwise_and(img,mask)
    return masked# used to manipulate 2 images. basically combine the two images

def process_img(image):
    original_image=image
    processed_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    processed_img=cv2.Canny(processed_img,threshold1=100,threshold2=200)
    # MAKE CHNAGES ACCORDINGLY . THE VERTICES ARE DIFFERENT FOR DIFFERENT GAMES .
    vertices=np.array([[10,500],[10,300],[300,200],[500,200],[800,300]],np.int32) # defining the vertices in the form of integer 2^31
    processed_img=roi(processed_img,[vertices])
    return processed_img

#def main():
last_time=time.time()
while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,100,800,700)))
    new_screen = process_img(screen)
    #new_screen=cv2.cvtColor(new_screen,cv2.COLOR_GRAY2RGB)
    last_time=time.time()
    cv2.imshow("window",new_screen)


    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows 
        break
