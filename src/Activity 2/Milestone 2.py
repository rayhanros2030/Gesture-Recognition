import cv2
import numpy as np


img = cv2.imread("../SampleImages/coins5.jpg")
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Original", img)
cv2.drawContours(gImg, contours, -1, (0, 255, 0), 2)
img2 = cv2.inRange(img, np.array([220, 50, 50]), np.array([255, 255, 255]))
result = cv2.bitwise_and(img, img, mask=img2)
cv2.imshow("In Range", img2)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Create a color image to draw contours on


# Draw contours i

# Display the results
cv2.imshow('Binary Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
img = cv2.imread("BallFinding/Blue1BG1Mid.jpg")
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Original", img)
img2 = cv2.inRange(img, np.array([220, 50, 50]), np.array([255, 255, 255]))
result = cv2.bitwise_and(img, img, mask=img2)
cv2.imshow("In Range", img2)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread("SampleImages/wildColumbine.jpg")
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(height, width, depth) = img.shape
# a rectangular mask that keeps the middle of a picture
rMask = np.zeros((height, width, 1), np.uint8)
rMask[50:height // 2, 50:width // 2] = 255
# applying a mask to an image (this is a bit weird)
maskedImg1 = cv2.bitwise_and(img, img, mask=rMask)
cv2.imshow("Original", img)
cv2.imshow("Rectangular Mask", maskedImg1)
# Using thresholding to create a mask
res, tMask = cv2.threshold(gImg, 128, 255, cv2.THRESH_BINARY)
maskedImg2 = cv2.bitwise_and(img, img, mask=tMask)
cv2.imshow("Threshold Mask", maskedImg2)
cv2.waitKey(0)
"""