import cv2
vidCap = cv2.VideoCapture(0)
for i in range(300):
    ret, img = vidCap.read()
    img2 = img[:, ::-1, :]
    cv2.imshow("Webcam", img2)
    cv2.waitKey(10)




cv2.destroyAllWindows()
vidCap.release()