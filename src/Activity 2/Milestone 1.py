import cv2
vidCap = cv2.VideoCapture(0)
Blurdir = 2
for i in range(300):
    ret, img = vidCap.read()
    img2 = img[:, ::-1, :]
    blurred_image = cv2.GaussianBlur(img2, (59 - Blurdir, 59 - Blurdir), 0)
    Blurdir += 2
    cv2.imshow("Webcam", blurred_image)
    cv2.waitKey(10)




cv2.destroyAllWindows()
vidCap.release()