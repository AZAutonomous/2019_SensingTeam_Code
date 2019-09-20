# import the necessary packages
from collections import deque
import numpy as np
import imutils
import cv2
import sys
import math
import time
from matplotlib import pyplot as plt

#Run masking or tracking
maskingOrTrack = False
useTracker = True

#Name of desired image
# Directory + name
imageString = "FilesToProcess/" + 'Red1.jpg'

#Masks: RedPart1, RedPart2, Yellow, Orange, Purple, Blue
masks = (((110,60,196),(129,255,255)), None)
#(((0,60,165),(24,255,255)), ((172,60,165),(180,255,255)),
#         ((24,60,165),(33,255,255)), ((0,60,165),(24,255,255)), ((113,60,165),(129,255,255)), ((106,60,165),(129,255,255)))

# Returns a mask
def getMask(imageHSV,pos, masks):

    # create NumPy arrays from the boundaries
    lower = np.array([masks[pos][0][0],masks[pos][0][1],masks[pos][0][2]], dtype="uint8")
    upper = np.array([masks[pos][1][0],masks[pos][1][1],masks[pos][1][2]], dtype="uint8")
    mask = cv2.inRange(imageHSV, lower, upper)
    return mask

# Converts a rgb image into hsv and applies filter
def process(imageIn, erodeOrNah, masks):

    # Converst the image to hsv
    imageHSV = cv2.cvtColor(imageIn,cv2.COLOR_BGR2HSV)

    # find the colors within the specified boundaries and apply
    # the mask
    mask = getMask(imageHSV, 0, masks)
    for iMasks in range(1,len(masks)):
        mask = cv2.bitwise_or(mask, getMask(imageHSV, iMasks, masks))
        
    
    if(not erodeOrNah):
       return mask
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    return cnts

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

# Used for creation of trackbar
def nothing(*arg):
    pass

def mask(masks, imageString):

        #Presets the trackbar
        print(masks[0])
        hsvVal = masks[0]
    
        #Creates the window to display
        cv2.namedWindow('RGB')
        
        #Creates the track bar
        cv2.createTrackbar('lower - h', 'RGB', hsvVal[0][0], 180, nothing)
        cv2.createTrackbar('lower - s', 'RGB', hsvVal[0][1], 255, nothing)
        cv2.createTrackbar('lower - v', 'RGB', hsvVal[0][2], 255, nothing)

        cv2.createTrackbar('upper - h', 'RGB', hsvVal[1][0], 180, nothing)
        cv2.createTrackbar('upper - s', 'RGB', hsvVal[1][1], 255, nothing)
        cv2.createTrackbar('upper - v', 'RGB', hsvVal[1][2], 255, nothing)

        # Forever loop
        while True:
            
            #Gets the hsv values from the track bar
            lh = cv2.getTrackbarPos('lower - h', 'RGB')
            ls = cv2.getTrackbarPos('lower - s', 'RGB')
            lv = cv2.getTrackbarPos('lower - v', 'RGB')
            uh = cv2.getTrackbarPos('upper - h', 'RGB')
            us = cv2.getTrackbarPos('upper - s', 'RGB')
            uv = cv2.getTrackbarPos('upper - v', 'RGB')

            #Sets the hsvVal to the track bar
            if(useTracker):
                masks = (((lh,ls,lv),(uh,us,uv)),((lh,ls,lv),(uh,us,uv)))
            
            # Reads in image and converts size
            imageRead = cv2.imread(imageString)
            imageRead = imutils.resize(imageRead, width=300, height=300)

            # find the colors within the specified boundaries and apply
            # the mask
            mask = process(imageRead, False, masks)
            output = cv2.bitwise_and(imageRead, imageRead, mask=mask)

            imageOut = np.hstack([imageRead, output])

            # Display the resulting frame
            cv2.imshow('RGB', imageOut)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                masking = False
                break

        # When everything done, release the capture
        cv2.destroyAllWindows()

def track(masks, imageString):

    #Reads the image
    imageRead = cv2.imread(imageString)

    #declares the number of sections making up one side (maybe should use squares)
    gridNum = 4

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
            cnts = process(image, True, masks)
            
            # only proceed if at least one contour was found
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # only proceed if the radius meets a minimum size
                if radius > 10:

                    #Image cropping code to get image
                    yStart, xStart, yEnd, xEnd = bound(height, width, colStart + x, rowStart + y, cropSize)
            
                    temp = imageRead[yStart:yEnd, xStart:xEnd]
                    cnts = process(temp, True, masks)

                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)

                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    #cv2.circle(temp, (int(x), int(y)), int(radius),
                    #           (0, 255, 255), 2)

                    #Checks if the image is unique in the already found images
                    unique = True
                    for goodImage in images:
                        if(math.sqrt((goodImage[0] - x)**2 + (goodImage[1] - y)**2) < radius*2):
                            unique = False
                            break
                    if(unique):
                        
                        #Re-crops the image
                        #gets the dimensions of the image
                        heightC, widthC, channelsC = temp.shape
                        yStart, xStart, yEnd, xEnd = bound(heightC, widthC, x, y, radius*4)
                        temp = temp[yStart:yEnd, xStart:xEnd]
                        
                        # Resizes the image
                        #temp =imutils.resize(temp, width=200, height=200);
                        # Saves as a found image
                        images.append((x , y, temp))
                        #Displays the image
                        cv2.namedWindow(str(gridRow) + "-" + str(gridCol))
                        cv2.imshow(str(gridRow) + "-" + str(gridCol), temp)

#cv2.imwrite("new_image" + "-" + str(count)+ ".jpg", image)


#Decides what part of the code to run
def main(masks, imageString, maskingOrTrack, useTracker):
    if(maskingOrTrack):
        track(masks, imageString)
    else:
        mask(masks, imageString)

main(masks, imageString, maskingOrTrack, useTracker)
