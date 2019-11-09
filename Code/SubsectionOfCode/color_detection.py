# import the necessary packages
import numpy as np
import cv2

#declare variables for the most common color and its number of pixels
modePixels = -1
modeColor = ''

#declares the dimension of the square grid map
gridDim = 6
#loads the image
image = cv2.imread('Blue3.jpg')

#defines the list of bgr color boundaries
ranges = [
    #([17, 15, 100], [50, 56, 200], 'Red'),
    #([0, 0, 0], [255, 255, 255], 'All'),
    #([86, 31, 4], [220, 88, 50], 'Blue'),
    #([50, 42, 5], [150, 70, 200], 'Broad blue'),
    ([190, 90, 62], [255, 255, 196], 'Yellow')
]

#gets the dimensions of the image
height, width, channels = image.shape

# loop over the boundaries
for (lower, upper, color) in ranges:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
 
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    #counts the number of pixels that aren't black
    colorPixels = cv2.countNonZero(mask)

    #checks for the most common color
    if (colorPixels > modePixels):
        modePixels = colorPixels
        modeColor = color
                 
    #show the masked image
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Most common color: " + modeColor)
print("Width: " + str(width))
print("Height: " + str(height))

#shows the masked image next to the original image
#cv2.imshow("images", np.hstack([image, output]))

#could have a variable representing "center of mass", then exclude pixels greater than x
#standard deviations away from the center of mass
