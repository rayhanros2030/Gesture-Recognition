import cv2
import numpy as np
# Read in image and display
face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_eye_tree_eyeglasses.xml")


faceImg = cv2.VideoCapture(0)

while True:
    ret, frame = faceImg.read()
    # Load trained model

    # Apply model to detect faces in grayscale version of the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw each rectangle, one by one, on the original image
    for (x, y, w, h) in face_rects: # eyeRects: #
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("Face", frame)
    cv2.waitKey()