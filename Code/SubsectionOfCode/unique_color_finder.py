#Author: Jonathan Goh
#Date: 10/19/2018
#Description: Divides the input image into a grid, identifies the most prevalent unique-colored pixel,
#identifies adjacent sections that exceed the threshold for percentage of unique-colored pixels, then
#returns the unique color and a cropped image of the subsections containing the unique-colored pixels

# import the necessary packages
import numpy as np
import cv2
import imutils

#declares the number of sections making up one side (maybe should use squares)
gridNum = 2

#specifies the minimum pixel percentage of a unique color needed to signal its presence in a grid
threshold = .10 #this will change depending on the image size, so a static value must be determined

#loads the image
image = cv2.imread('y5.jpg')
image = imutils.resize(image, width=600, height = 600)

#gets the dimensions of the image
height, width, channels = image.shape

#calculates grid dimensions 
gridWidth = width / gridNum
gridHeight = height / gridNum

#print image dimensions
print("Width of image: " + str(width))
print("Height of image: " + str(height))
print("Grid Width: " + str(gridWidth))
print("Grid Height: " + str(gridHeight))

#declare variables for the most common color and its number of pixels
modePixels = -1
modeColor = ''

#defines the list of bgr color boundaries
ranges = [
    ([20, 0, 100], [100, 100, 255], 'Red'),
    ([0, 100, 200], [75, 168, 255], 'Orange'), 
    ([0, 168, 200], [75, 255, 255], 'Yellow'),
    #([0, 195, 200], [75, 255, 255], 'Green'), #may not need because not unique
    #([50, 0, 0], [255, 230, 80], 'Blue'), #may not need because not unique
    ([100, 0, 50], [255, 175, 200], 'Violet') #may want to redefine, especially in the magenta by red
    
]

#identifies each grid
for i in range(gridNum):
    for j in range(gridNum):
        #defines corners of the grid
        rowStart = round(gridHeight * i)
        rowEnd = round(gridHeight * (i + 1))
        colStart = round(gridWidth * j)
        colEnd = round(gridWidth * (j + 1))

        #creates an image for the grid piece
        cropped = image[rowStart:rowEnd, colStart:colEnd]

        #loop over the boundaries
        for (lower, upper, color) in ranges:
            #create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            
            #find the colors within the specified boundaries and apply the mask
            mask = cv2.inRange(cropped, lower, upper)
            output = cv2.bitwise_and(cropped, cropped, mask = mask)

            #counts the number of pixels that aren't black
            colorPixels = cv2.countNonZero(mask)

            #divides the number of colored pixels by the number of pixels in the grid
            pixelPercentage = colorPixels / (gridHeight * gridWidth)
            
            #determines if the percentage of colored pixels is greater than the threshold
            if pixelPercentage > threshold:
                print("Grid: (" + str(i) + "," + str(j) + ") exceeds the " + color + " threshold" + str(pixelPercentage))
            
            #if the threshold is exceeded, the coordinates of the grid are stored in an array
            
            #shows the masked image next to the original image
            cv2.imshow("Grid: (" + str(i) + "," + str(j) + ") - Color: " + color, np.hstack([cropped, output]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #checks for the most common color
            if (colorPixels > modePixels):
                modePixels = colorPixels
                modeColor = color
                
        #prints the grid and its most common color
        print("Grid: (" + str(i) + "," + str(j) + ") - Most common color: " + modeColor)



        
