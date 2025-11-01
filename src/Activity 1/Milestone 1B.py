import cv2
import os

imageNames = os.listdir("../SampleImages")
for names in imageNames:
    if names.endswith('jpg') or names.endswith('png'):
        images = cv2.imread("SampleImages/" + names)
        cv2.imshow("Images", images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()