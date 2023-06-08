# SENTDEXs lane detection code . based on simple opencv and linear algebra . Not that accurate .

import time
from statistics import mean

import cv2
import numpy as np  # Numpy is a mathematical library for Python that adds support for large, multi-dimensional arrays and matrices and a large collection of high-level mathematical functions to operate on these arrays.
from numpy import (  # ones- creates a narray of ones of any given shape and size ; vstack - vertically stacks the array
    ones, vstack)
from numpy.linalg import lstsq
from PIL import ImageGrab

from directkeys import A, D, PressKey, ReleaseKey, S, W


def roi(img, vertices):
    
     #blank mask:
     mask = np.zeros_like(img)   
    
     #filling pixels inside the polygon defined by "vertices" with the fill color    
     cv2.fillPoly(mask, vertices, 255)
    
#     #returning the image only where mask pixels are nonzero
     masked = cv2.bitwise_and(img, mask)
     return masked


def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some random line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  # 
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)# min y is the furthest up in the horizon , the top of the slope
        max_y = 800 # maximum slope line we can have 
        new_lines = []
        line_dict = {} # creating an empty dictionary

        for idx,i in enumerate(lines): # going line by line ; idx- index ; i- line
            for xyxy in i: # xyxy- 2 coordinates in line . x1y1x2y2
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                # CALCULATING THE LINE .
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3]) # https://www.mathsisfun.com/algebra/line-equation-2points.html
                A = vstack([x_coords,ones(len(x_coords))]).T # basically creating a vertical stack of array1 - x_coords , array2- ones(len(x_coords)): which is creating  an array filled with ones of length x_coords , done to match the size of the 2 arrays . and then finally .T transposes the array.
                m, b = lstsq(A, y_coords)[0] # returns a least square solution to a matrix equation . it returns a tuple so [0] is basically taking the first value of the tuple. https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.lstsq.html#
                # m and b are slope . 

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m 
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]] # the dictionary stores the slope bias and the actual x and y values. basically according to the idex of the lines assign x and y values for the 2 coordinates
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy() # we area making a copy coz they were multiple edges were found when we used cv2.Canny , and Hough lines are based on those edges . so if there are multiple copies of the same line on the edge we dont consider that in our lane line . 
            # we do this because as we saw in Hough lines . they were many lines detected so we wanna take the lane line that is nearest to the slope .
            m = line_dict[idx][0] # taking the first value in the tuple line_dict 
            b = line_dict[idx][1] # taking the 2nd value in line_dict tuple as slope 
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0: # if there is nothing in the dictionary final_lanes . then the lane is m,b,line. 
                final_lanes[m] = [ [m,b,line] ]
                
            else: # else if there are lines iterated in the list then...
                found_copy = False # set this as false

                for other_ms in final_lanes_copy: # for other slopes found in lanes copy i.e other lines found

                    if not found_copy: # if found_copy is True then the below formula is to pick the best possible slope line for our lane .
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else: # if it doesn't satisfy the above equation then go with the default 1st,2nd and 3rd slope line i.e the most common slope lines . 
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {} # creating a dictionary to find the average slope line.

        for lanes in final_lanes: 
            line_counter[lanes] = len(final_lanes[lanes]) # finding the length of the lanes

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2] # sorting the lenght of the lanes in a dictionary (.item()- arranges the array items in a dictionary) and taking the top 2 lanes 

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) # returning the average x1,y1,x2,y2 values

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2] # returns the coordinates of lane 1 and lane 2 
    except Exception as e: # returns an error if no lines are detected
        print(str(e))


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    
    vertices = np.array([[130,540],[410,350],[570,350],[915,540]], np.int32)

    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:        
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,      20,       15)
    try:
        l1, l2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image



def main():
    last_time = time.time()
    while True:
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image = process_img(screen)
        cv2.imshow('window', new_screen)
        cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()