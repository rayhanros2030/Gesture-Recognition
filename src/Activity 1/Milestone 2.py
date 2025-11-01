import cv2

img1 = cv2.imread("../SampleImages/antiqueTractors.jpg")
ROI1 = img1[250:550, 570:860, :]
img2 = cv2.imread("../SampleImages/grandTetons.jpg")
ROI2 = img2[250:550, 570:860, :]
img = cv2.addWeighted(img1, 0.6, img2, 0.4, 0.3)
cv2.imshow("Combination", img)
cv2.waitKey(0)