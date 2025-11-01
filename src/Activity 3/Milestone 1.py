import os
import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt # Used to display images in Colab



# This line of code is used to plot points in a graph

def plotDataAndPoint(data, categ, newPoints):
    # Takes in data and a list of responses, as well as a list
    # of new points, and it creates a scatter plot
    # data where categ is 0 are red
    # data where categ is 1 are blue
    # new data is green
    red = data[categ==0] # this selects just data where bool is true
    plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
    blue = data[categ==1]
    plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
    # Plot a new point in green
    plt.scatter(newPoints[:, 0], newPoints[:, 1], 80, 'g', 'o')
    plt.show()

trainData = np.random.randint(0, 100, (40, 2)).astype(np.float32)

mins = trainData.min(axis = 1)

atLeast25 = mins >= 25
atMost75 = mins < 75

inBounds = np.logical_and(atLeast25, atMost75)
# Generate a new point
newPoint = np.random.randint(0, 100, (1, 2)).astype(np.float32)
# Plot random data, and new (random) point
categs = inBounds.astype(np.float32)

plotDataAndPoint(trainData, categs, newPoint)



knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, categs)

# Report category of new point (based on k=3)
ret, result, neighbors, dist = knn.findNearest(newPoint, 100)
print("result:", result)
print("neighbors:", neighbors)
print("distances:", dist)
print("Min value", mins)