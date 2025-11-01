import cv2

img1 = cv2.imread("../SampleImages/snowLeo1.jpg")
cv2.imshow("Leopard 1", img1) # This is basically a way of printing out images
cv2.waitKey(0) # tells the program to wait indefinitely until a user presses a key
cv2.destroyAllWindows() # until the key has been pressed, the images will remain
img2 = cv2.imread("../SampleImages/beachBahamas.jpg")
cv2.imshow("The Beach", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
img3 = cv2.imread("../SampleImages/landscape1.jpg")
cv2.imshow("Landscape", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
img4 = cv2.imread("../SampleImages/wildColumbine.jpg")
cv2.imshow("Porcupine", img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

