import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Test.png',0)
edges = cv2.erode(img, None, iterations=2)
edges = cv2.Canny(edges,100,200)
edges = cv2.dilate(edges, None, iterations=2)
    
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
