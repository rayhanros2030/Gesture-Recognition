import cv2
import numpy as np # common way to import numpy
import random

vidCap = cv2.VideoCapture(0)
for i in range(300):
    ret, img = vidCap.read()
    img2 = img[:, ::-1, :]
    (rows, cols, dep) = img.shape
    transMatrix = np.float32([[4, 0, random.randint(0,100)], [0, 6, random.randint(0,100)]])  # change 30 and 50
    transImag = cv2.warpAffine(img, transMatrix, (cols, rows))
    cv2.imshow("Webcam", transImag)
    cv2.waitKey(10)