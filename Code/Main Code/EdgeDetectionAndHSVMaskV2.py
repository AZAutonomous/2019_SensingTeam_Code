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

# Name of desired image
# Directory + name
imagePath = "D:\\School\\AZA\\2019_SensingTeam_Code\\CroppedImages\\" # Path from Desktop, where the code is ran
imageName = "Test3.jpg"


# For debugging, press space to look through the filters that is occuring and esc to go to next grid partition
# Only uses value when something is found on grid partition
debug = True # this tells the code wether to output the current image in a graphics window

# Grid Size
gridNum = 1

# Masks: RedPart1, RedPart2, Yellow, Orange, Purple, Blue, Green is currently disabled
masks = (((0,60,165),(24,255,255)), ((172,60,165),(180,255,255)),
         ((24,60,165),(33,255,255)), ((0,60,165),(24,255,255)), ((113,60,165),(129,255,255)),
         ((110,60,220),(129,255,255)))#, ((95,60,202),(102,255,255)))

# before the sensors are integrated,
# we have globals to simulate each kinematic datum
pixhawk = None # this is the pointer to the sensor data will go

pitch = 0.0
roll = 0.0
yaw = 0.0
latitude = 0.0
longitude = 0.0
altitude = 0.0
heading = 0.0


def getPos():
    """
    This function populates the globals for position data
    """
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
    """
    This function populates the 3-D attitude globals
    """
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
    """
    This function tells the image processor to wait until a key is pressed
    Each key tells the function what to reassign the global debug variable
    """
    k = -1
    if(debug):
        cv2.imshow('debug', imageIn)
        print(message) # tell the user what is happening in the console window
        while(1):
            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop processing
                return False
            elif k==32: # Space key will cycle to the next image processing func
                return True
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            else:
                print("Please press space to continue")
    # if debug is false to begin with we have to keep it as false
    return False


# Returns a mask
def getMask(imageHSV, pos, masks):
    """
    This function returns a mask
    """
    # create NumPy arrays from the boundaries
    lower = np.array([masks[pos][0][0], masks[pos][0][1], masks[pos][0][2]], dtype = "uint8")
    upper = np.array([masks[pos][1][0], masks[pos][1][1], masks[pos][1][2]], dtype = "uint8")
    mask = cv2.inRange(imageHSV, lower, upper)
    return mask


# Converts a rgb image into hsv and applies filter
def processHSV(imageIn, debug, iMask, masks):
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
    maskHSV = getMask(imageHSV, iMask, masks)
    debug = myWaitKey(maskHSV, debug, "HSV Mask")

    # Remove noise from the color filtered image
    maskHSV = cv2.erode(maskHSV, None, iterations=2)
    debug = myWaitKey(maskHSV, debug, "HSV erode")
    maskHSV = cv2.dilate(maskHSV, None, iterations=2)
    debug = myWaitKey(maskHSV, debug, "HSV dilate")

    #Destroy windows if debug mode turned on and return contours and the mask
    cv2.destroyWindow('debug')
    return maskHSV


def processCanny(imageIn, debug):

    #Creates the kernel for the smoothing filter and smooths the image
    debug = myWaitKey(imageIn, debug, "Start edge detection")
    #kernel = np.ones((5,5),np.float32)/25
    #imageIn = cv2.filter2D(imageIn,-1,kernel)
    #debug = myWaitKey(imageIn, debug, "Edge smooth")

    # Remove noise by blurring with a Gaussian filter
    imageBlur = cv2.GaussianBlur(imageIn, (3, 3), 0)
    debug = myWaitKey(imageBlur, debug, "Edge Blur")

    # converting to gray scale
    gray = cv2.cvtColor(imageIn, cv2.COLOR_BGR2GRAY)
    debug = myWaitKey(gray, debug, "gray")

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)
    debug = myWaitKey(img, debug, "Gaussian Blur")

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    debug = myWaitKey(laplacian, debug, "laplacian")
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    debug = myWaitKey(sobelx, debug, "sobelx")
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y
    debug = myWaitKey(sobely, debug, "sobely")


    #Finds the edges from the image
    mask = cv2.Canny(img, 100, 200)
    debug = myWaitKey(mask, debug, "Edge Canny")

    # Closes possible unfilled shapes
    mask = cv2.dilate(mask, None, iterations=1)
    debug = myWaitKey(mask, debug, "Edge dilate")

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

    return mask


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


def blobDetection(mask):

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;

    # Filter by Area.
    params.filterByArea = True
    params.maxArea = 10000
    params.minArea = 1000

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask=255-mask
    keypoints = detector.detect(reversemask)
    if keypoints:
        if len(keypoints) > 4:
            # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints=keypoints[0:3]

    return keypoints


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

            #Loops throgh different HSV bounds that were created
            allKeyPoints = []
            for iMask in range(0,len(masks)):

                # Get coordinate images for cropped image section
                center = None
                mask = processHSV(image, True, iMask, masks) #we changed True from false to see hsv

                #Retrives all blobs from the HSV mask
                keypoints = blobDetection(mask)

                #Adds the keypoints to the total keypoints
                allKeyPoints += keypoints

            #Processes via canny detection
            mask = processCanny(image, debug)

            # Draw green circles around detected blobs
            print("found %d blobs" % len(allKeyPoints))
            im_with_keypoints = cv2.drawKeypoints(image, allKeyPoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #Prints information of the valid keypoints
            for keypoint in allKeyPoints:
                print("(" + str(keypoint.pt[0]) + "," + str(keypoint.pt[1]) + ")" + " size = "+ str(keypoint.size))
            debug = myWaitKey(im_with_keypoints, debug, "Blob detection")

            #

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

    # These funcs will either init the avionics data or give them
    # the default values if pixhawk is not connected

    getPos()
    getAttitude()
    count = 0

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


# this code calls main
print(imagePath)
main(imagePath, imageName, debug, gridNum, masks)