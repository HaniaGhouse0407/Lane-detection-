import time

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 1)
    except:
        pass


def roi(img,vertices):
    mask=np.zeros_like(img)
    cv2.fillConvexPoly(mask,vertices,255)
    masked=cv2.bitwise_and(img,mask)
    return masked

def process_img(image):
   original_image=image
   processed_img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   processed_img=cv2.Canny(processed_img, threshold1=100,threshold2=100)

   vertices=np.array([[50,270], [220,160], [360,160], [480,270]],np.int32)
   processed_img=cv2.GaussianBlur(processed_img,(5,5),0)
   #processed_img=cv2.cvtColor(processed_img,cv2.COLOR_BGR2HSV)
   lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, 20, 10)# makes lines after it detects edges. 
   processed_img=roi(processed_img,[vertices])
   draw_lines(processed_img,lines)
   return processed_img

last_time=time.time()
while(True):
    screen =np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    new_screen=process_img(screen)
    print("loop took {} seconds".format(time.time()-last_time))
    last_time=time.time()
    cv2.imshow("window",new_screen)



    if cv2.waitKey() & 0xFF == ord("q"):
        cv2.destroyAllWindows
        break