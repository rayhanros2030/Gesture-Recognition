import cv2
import numpy as np

# Read the image
image = cv2.imread('../BallFinding/Blue1BG1Mid.jpg')  # Replace with your image path

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Method 1: Basic Thresholding
# Pixels > 127 become 255 (white), others become 0 (black)

ret, thresh_basic = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
maskedImg2 = cv2.bitwise_and(image, image, mask=thresh_basic)
contrs, hier = cv2.findContours(thresh_basic, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contrs, -1, (0,255,0), 2)
for cntr in contrs:
    print(cv2.contourArea(cntr))
    (ulx, uly, wid, hgt) = cv2.boundingRect(cntr)
    cv2.rectangle(image, (ulx, uly), (ulx + wid, uly + hgt), (0, 0, 0), 2)
    convHull = cv2.convexHull(cntr)
    cv2.drawContours(image, [convHull], -1, (255, 255, 0), 1)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()