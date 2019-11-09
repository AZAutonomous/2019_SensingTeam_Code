# baseline cnn model for mnist
#from numpy import mean
#from numpy import std
#from matplotlib import pyplot

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from skimage import io
import numpy as np
import csv
import os
from sklearn.utils import shuffle
from numpy.random import seed
seed(1)

#CNN pre-variables
NUM_ROWS = 100
NUM_COLS = 300
NUM_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 10

#Classication location in path
csvFileLocaiton = "D:\\HonorsThesis\\blink\\BCA\\Blink_Comparison_Data\\Callie1"
csvFileName = "Callie1_class.csv"
fullCSVFilePath = csvFileLocaiton + "/" + csvFileName

#Opens the classication file
imageClassications = []
with open(fullCSVFilePath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    csvLines = 0
    for row in csv_reader: # each row is a list
        imageClassications.append(row)
        csvLines+=1

# create an array where we can store our 10 pictures
myImages = []
# and the correct values
myLabels = []


# Reads all the images
path = 'D:\\HonorsThesis\\blink\\BCA\\Blink_Comparison_Data\\Callie1\\cropped_frames'

#Loops through all the images in the csv file given
eyesvsclosed = [0,0];
labelIndexToNumber = [];
for row in imageClassications:
        if(int(row[2]) == 1 or int(row[2]) == 2):
            image = io.imread(os.path.join(path, row[0] + "_crop.jpg"))
            myImages.append(image)
            
            labelIndexToNumber.append([row[2],row[0]])
            #Appends 0 for open and 1 for closed eye position
            if(int(row[2]) == 1):
                myLabels.append(0)
                eyesvsclosed[0] = eyesvsclosed[0] + 1
            else:#must be closed because classification is either 1 or 2 at this point
                myLabels.append(1)
                eyesvsclosed[1] = eyesvsclosed[1] + 1

print("Ratio of eyes open to eyes closed is:")
print(eyesvsclosed)
print("Amount of images is:")
print(len(myImages))
print("Amount of label is:")
print(len(myLabels))

myImages = np.array(myImages)
myImages = myImages.reshape((myImages.shape[0], 100, 300, 3))
myLabels = to_categorical(myLabels)

# define cnn model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax')) #change this from sigmoid to ->
#
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Fold amount
foldAmount = 1

# Makes the data divisible by the foldAmount, so we could split it
myImages = myImages[0:myImages.shape[0] - (myImages.shape[0]%foldAmount)]
myLabels = myLabels[0:myLabels.shape[0] - (myLabels.shape[0]%foldAmount)]

# Splits the data in train and test data
trainX, trainY, testX, testY = myImages[:-1000], myLabels[:-1000], myImages[-1000:], myLabels[-1000:]
scores, histories = list(), list()
    

# Loops through the folds and gets the accuracy
# fit fold
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
    
# evaluate model
temp, acc = model.evaluate(testX, testY, verbose=0)
myPredict = model.predict_(testX, verbose=0)

#stores scores
scores.append(acc)
histories.append(history)

print("Scores is:")
print(scores)
print("histories is:")
print(histories)
print("temp is:")
print(temp)

#CONFUSION MATRIX SETUP
#True positives, Fasle Positives
#False Negatives, Positive Negative
#trainX, trainY, testX, testY = myImages[:-1000], myLabels[:-1000], myImages[-1000:], myLabels[-1000:]
#
##fit model
#history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
## evaluate model
#_, acc = model.evaluate(testX, testY, verbose=0)
#print('> %.3f' % (acc * 100.0))
# =============================================================================
