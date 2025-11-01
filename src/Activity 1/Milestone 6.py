import cv2
import numpy as np # common way to import numpy
import random

vidCap = cv2.VideoCapture(0)
for i in range(300):
    ret, img = vidCap.read()
    img2 = img[:, ::-1, :]
    (rows, cols, dep) = img.shape
    rotMat = cv2.getRotationMatrix2D((cols / 2, rows / 2), i, 1)
    rotImg = cv2.warpAffine(img, rotMat, (cols, rows))
    cv2.imshow("Rotated", rotImg)
    cv2.waitKey(10)