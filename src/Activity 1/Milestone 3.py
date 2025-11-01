import cv2
import numpy as np

Image = cv2.imread("../SampleImages/mightyMidway.jpg")
cv2.line(Image, (250, 100), (250, 300), (0, 0, 255))
cv2.line(Image, (250, 160), (200, 220), (0, 0, 255))
cv2.line(Image, (250, 160), (300, 220), (0, 0, 255))
cv2.line(Image, (250, 300), (210, 360), (0, 0, 255))
cv2.line(Image, (250, 300), (290, 360), (0, 0, 255))
cv2.circle(Image, (250, 100), 30, (220, 0, 0), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(Image, "Hi, there", (10, 270), font, 1, (255, 255, 255))
cv2.imshow("White", Image)
cv2.imwrite("../SampleImages/whitePic.jpg", Image)
cv2.waitKey(0)
cv2.destroyAllWindows()