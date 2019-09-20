# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob

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

blobsNotFound = []
image = "Test1.jpg"

orig_img = cv2.imread(image)   
debug = myWaitKey(orig_img, True, "Orig Image")

# Blur image to remove noise
frame=cv2.GaussianBlur(orig_img, (3, 3), 0)
debug = myWaitKey(frame, True, "Gaussian Blur")

# Switch image from BGR colorspace to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
debug = myWaitKey(hsv, True, "HSV")

# define range of purple color in HSV
yellowMin = (24,60,165)
yellowMax = (33,255,255)

# Sets pixels to white if in purple range, else will be set to black
mask = cv2.inRange(hsv, yellowMin, yellowMax)
debug = myWaitKey(mask, True, "Mask")

# Bitwise-AND of mask and purple only image - only used for display
res = cv2.bitwise_and(frame, frame, mask= mask)

#    mask = cv2.erode(mask, None, iterations=1)
# commented out erode call, detection more accurate without it

# dilate makes the in range areas larger
mask = cv2.dilate(mask, None, iterations=1)    

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 30
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
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
    print("found %d blobs" % len(keypoints))
    if len(keypoints) > 4:
        # if more than four blobs, keep the four largest
        keypoints.sort(key=(lambda s: s.size))
        keypoints=keypoints[0:3]
else:
    print("no blobs")

# Draw green circles around detected blobs
im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
# open windows with original image, mask, res, and image with keypoints marked
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)     
cv2.imshow("Keypoints", im_with_keypoints)            
