# import the necessary packages
from collections import deque
import numpy as np
import imutils
import cv2
import sys
import math
import time
from matplotlib import pyplot as plt
import os
import shutil
from pymavlink import mavutil
import threading
import serial
import json
import time

#Name of desired image
# Directory + name
imagePath = "" #Path from Desktop, where the code is ran
imageName = "Test1.jpg"


#For debugging, press space to look through the filters that is occuring and esc to go to next grid partition
#Only uses value when something is found on grid partition
debug = True

#Grid Size
gridNum = 1

#Masks: RedPart1, RedPart2, Yellow, Orange, Purple, Blue, Green <-Disabled
masks = (((0,60,165),(24,255,255)), ((172,60,165),(180,255,255)),
         ((24,60,165),(33,255,255)), ((0,60,165),(24,255,255)), ((113,60,165),(129,255,255)),
         ((110,60,220),(129,255,255)))#, ((95,60,202),(102,255,255)))


pixhawk = None

pitch = 0.0
roll = 0.0
yaw = 0.0
latitude = 0.0
longitude = 0.0
altitude = 0.0
heading = 0.0

def getPos():
    global latitude
    global longitude
    global altitude
    global heading
    threading.Timer(0.2, getPos).start()
    try:
        posMsg = pixhawk.recv_match(type="GLOBAL_POSITION_INT", blocking=False, timeout=10.0)
        latitude = posMsg.lat
        longitude = posMsg.lon
        altitude = posMsg.alt
        heading = posMsg.hdg
    except:
       pass

def getAttitude():
    global pitch
    global roll
    global yaw
    threading.Timer(0.2, getAttitude).start()

    try:
        attMsg = pixhawk.recv_match(type="ATTITUDE", blocking=False, timeout=10.0)
        pitch = attMsg.pitch
        roll = attMsg.roll
        yaw = attMsg.yaw
    except:
        pass

# Wait for key code
def myWaitKey(imageIn, debug, message):

    k = -1
    if(debug):
        cv2.imshow('debug', imageIn)
        print(message)
        while(1):
            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                return False
            elif k==32:
                return True
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            else:
                print(k) # else print its value
    return False


# Returns a mask
def getMask(imageHSV,pos, masks):

    # create NumPy arrays from the boundaries
    lower = np.array([masks[pos][0][0],masks[pos][0][1],masks[pos][0][2]], dtype="uint8")
    upper = np.array([masks[pos][1][0],masks[pos][1][1],masks[pos][1][2]], dtype="uint8")
    mask = cv2.inRange(imageHSV, lower, upper)
    return mask

# Converts a rgb image into hsv and applies filter
def process(imageIn, debug, fillInFilter, masks):
    global latitude
    global longitude
    global altitude
    global heading
    global pitch
    global roll
    global yaw
    
    # Converst the image to hsv, allows you to skip debug for specific image
    debug = myWaitKey(imageIn, debug, "Original Image")

    # Blur image to remove noise
    imageBlur = cv2.GaussianBlur(imageIn, (3, 3), 0)
    debug = myWaitKey(imageBlur, debug, "Gaussian Blur")
        
    imageHSV = cv2.cvtColor(imageBlur,cv2.COLOR_BGR2HSV)
    debug = myWaitKey(imageHSV, debug, "HSV Conversion")
    
    # find the colors within the specified boundaries and apply
    # the mask
    maskHSV = getMask(imageHSV, 0, masks)
    for iMasks in range(1,len(masks)):
        maskHSV = cv2.bitwise_or(maskHSV, getMask(imageHSV, iMasks, masks))
    debug = myWaitKey(maskHSV, debug, "HSV Mask")
    
    # Remove noise from the color filtered image
    maskHSV = cv2.erode(maskHSV, None, iterations=2)
    debug = myWaitKey(maskHSV, debug, "HSV erode")
    maskHSV = cv2.dilate(maskHSV, None, iterations=2)
    debug = myWaitKey(maskHSV, debug, "HSV dilate")
    
    #Creates the kernel for the smoothing filter and smooths the image
    debug = myWaitKey(imageIn, debug, "Start edge detection")
    #kernel = np.ones((5,5),np.float32)/25
    #imageIn = cv2.filter2D(imageIn,-1,kernel)
    #debug = myWaitKey(imageIn, debug, "Edge smooth")

    # Remove noise by blurring with a Gaussian filter
    imageBlur = cv2.GaussianBlur(imageIn, (3, 3), 0)
    debug = myWaitKey(imageBlur, debug, "Edge Blur")

    # [convert_to_gray]
    #imageGray = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
    #debug = myWaitKey(imageGray, debug, "Lapl Gray")
    
    # [laplacian]
    # Apply Laplace function
    #maskEhh = cv2.Laplacian(imageGray, cv2.CV_16S, 3)
    #debug = myWaitKey(maskEhh, debug, "Lapl Mask")
     
    # Filters the image and gets a mask
    mask = cv2.erode(imageBlur, None, iterations=3)
    debug = myWaitKey(mask, debug, "Edge erode")
    mask = cv2.Canny(mask, 100, 200)
    debug = myWaitKey(mask, debug, "Edge Canny")
    mask = cv2.dilate(mask, None, iterations=1)
    debug = myWaitKey(mask, debug, "Edge dilate")
    

    if(fillInFilter):
        #Creates a temporary copy of mask and a mask with zeros
        maskCopy = mask.copy()
        mask2 = np.zeros((np.array(mask.shape)+2), np.uint8)
        
        #Fills in the area that are not enclosed
        cv2.floodFill(mask, mask2, (0,0), (255))
        debug = myWaitKey(mask, debug, "Flood fill")
        
        #Gets rid of noise and removes non-closed shapes from mask
        mask = cv2.erode(mask, np.ones((3,3)))
        debug = myWaitKey(mask, debug, "Flood erode")
        mask = cv2.bitwise_not(mask)
        debug = myWaitKey(mask, debug, "Flood not")
        mask = cv2.bitwise_and(mask,maskCopy)
        debug = myWaitKey(mask, debug, "Flood and")
        
    # Combines color mask and edge detection mask
    mask = cv2.bitwise_or(maskHSV, mask)
    debug = myWaitKey(mask, debug, "HSV + Edge")
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    #Destroy windows if debug mode turned on and return contours and the mask
    cv2.destroyWindow('debug')
    return (cnts, mask)

def bound(height, width, x, y, distance):
    yStart = round(y - distance)
    xStart = round(x - distance)
    yEnd = round(y + distance)
    xEnd = round(x + distance)

    #Bounding code
    if(xStart < 0):
        xStart = 0
    if(yStart < 0):
        yStart = 0
    if(xEnd > width):
        xEnd = width
    if(yEnd > height):
        yEnd = height;

    return yStart, xStart, yEnd, xEnd

def track(imagePath, imageName,debug, gridNum, masks, count):
    
    #Reads the image
    imageRead = cv2.imread(imagePath + imageName)

    #gets the dimensions of the image
    height, width, channels = imageRead.shape

    #calculates grid dimensions 
    gridWidth = width / gridNum
    gridHeight = height / gridNum

    #Size of partition of cropped images when something found
    cropSize = round((gridWidth + gridHeight)/4)

    # row = height = y
    # col = width = x
    
    #Found images
    images = [];
    verCat = None
    for gridRow in range(gridNum):
        horCat = None
        for gridCol in range(gridNum):
                
            #defines corners of the grid
            rowStart = round(gridHeight * gridRow)
            rowEnd = round(gridHeight * (gridRow + 1))
            colStart = round(gridWidth * gridCol)
            colEnd = round(gridWidth * (gridCol + 1))
            
            #creates an image for the grid piece
            image = imageRead[rowStart:rowEnd, colStart:colEnd]

            # Get coordinate images for cropped image section
            center = None
            (cnts, mask) = process(image, False, False, masks)
                
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                # Values are relative to the cropped image from the grid
                ((xToGrid, yToGrid), radiusToGrid) = cv2.minEnclosingCircle(c)
                # Absolute values to the original image
                xAbs = colStart + xToGrid;
                yAbs = rowStart + yToGrid;
                
                # only proceed if the radius meets a minimum size
                if radiusToGrid > 10:

                    #Image cropping code to get image
                    yStart, xStart, yEnd, xEnd = bound(height, width, xAbs, yAbs, cropSize)
                    reCroppedImage = imageRead[yStart:yEnd, xStart:xEnd]

                    # Calculates how much the bound actually changed
                    zoom_out_up = yAbs - yStart;
                    zoom_out_left = xAbs - xStart;
                    
                    #For debugging purposes
                    if(debug):
                        (cnts, mask) = process(reCroppedImage, True, True, masks)
                    else:
                        (cnts, mask) = process(reCroppedImage, False, True, masks)

                    #Quit out if cnts is empty after recrop procress, uses fillShape masking
                    if not (len(cnts) > 0):
                        continue

                    #Relative value based on shift of contour
                    c = max(cnts, key=cv2.contourArea)
                    ((xRel, yRel), radiusRel) = cv2.minEnclosingCircle(c)

                    # Calculates the new centerValues
                    yAbsCenter = yAbs - zoom_out_up + round(yRel)
                    xAbsCenter = xAbs - zoom_out_left + round(xRel) 
                    
                    #Debugging, allows checking of object sizes
                    if(debug):
                        print("Radius is " + str(radiusRel))
                        print("Pos of Image : (" + str(xAbsCenter) + "," + str(yAbsCenter) + ")")
                        
                    #Re-crops the image
                    #gets the dimensions of the image
                    #if the radius of the image is good enough and is unique
                    if((radiusRel <= 36) and (radiusRel >= 13)):

                        #Checks if the image is unique in the already found images
                        unique = True
                        for goodImage in images:
                            # Distance calculation from the goodImage, absolute value comparison
                            if(math.sqrt((goodImage[0] - (xAbsCenter))**2 + (goodImage[1] - (yAbsCenter))**2) < radiusRel*2):
                                unique = False
                                break

                        #Skip the grid if not unique
                        if(not unique):
                            continue
                        
                        yStart, xStart, yEnd, xEnd = bound(height, width, (xAbsCenter), (yAbsCenter), radiusRel*3)
                        reCroppedImage = imageRead[yStart:yEnd, xStart:xEnd]
                        
                        # Resizes the image
                        # Saves as a found image
                        images.append((xAbsCenter, yAbsCenter, reCroppedImage))
                        
                        #Displays the imageString
                        if(debug):
                            cv2.imshow(str(gridRow) + "-" + str(gridCol) + " image", reCroppedImage)
                            #cv2.imshow(str(gridRow) + "-" + str(gridCol) + " mask", mask)
                            print( "pitch =" + str(pitch))
                            print( "roll =" + str(roll))
                            print( "yaw =" + str(yaw))
                            print( "latitude =" + str(latitude))
                            print( "longitude =" + str(longitude))
                            print( "altitude =" + str(altitude))
                            print( "heading =" + str(heading))
                            
                        else:
                            
                            #Creates the path to save the images with the image count
                            savepath = imagePath + "Preprocessed/" + str(count) + "_"

                            #Saves a cropped image of an area of interest
                            cv2.imwrite(savepath + "img.jpg", reCroppedImage)

                            # Data that will be saved
                            jsonPacket = {
				"pitch": pitch,
				"roll": roll,
				"yaw": yaw,
				"latitude": latitude,
				"longitude": longitude,
				"altitude": altitude,
				"heading": heading
                            }

                            # Writes a json file of the data of the image
                            with open(savepath + "data.json", 'w') as outfile:
                            	json.dump(jsonPacket, outfile)

                            # Increments to the next image count
                            count+=1
                            
                            
    return count            


#Decides what part of the code to run
def main(imagePath, imageName,debug, gridNum, masks):

    print(imagePath)
    global pixhawk

    # Incremental for the image counter
    count = 0
    
    # Specifies the pixhawk dialect to the mavlink
    mavutil.set_dialect("ardupilotmega")

    # Establishes connection to pixhawk
    try:                                      #Port pixhawk connects to
        pixhawk = mavutil.mavlink_connection("COM4" ,autoreconnect=True)
    except:
        print("Could not connect")


    getPos()
    getAttitude()
    
    if(not debug):
        
        #Removes old preprocessed files
        if(os.path.exists(imagePath + "Preprocessed/")):
            shutil.rmtree(imagePath + "Preprocessed/")

        #Makes a folder to put preprocessed files into
        os.mkdir(imagePath + "Preprocessed/")

        #Loops through all the images in the folder
        print(imagePath)
        for f in os.listdir(imagePath):
            if f.lower().endswith(".jpg"):
            
                #Save the previous count to calculate how many output files are written
                tempCount = count

                 #Returns a new count and outputs json file with images of tracked images
                start_time = time.time()
                count = track(imagePath, f, debug, gridNum, masks, count)
                elpased_time = time.time() - start_time

                #Calculates output files written and prints it to screen
                print("Added " + str(count - tempCount) + " files in " + str(round(elpased_time,2)) + "seconds")
    else:
        track(imagePath, imageName,debug, gridNum, masks, count)

print(imagePath)
main(imagePath, imageName, debug, gridNum, masks)
