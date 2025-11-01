import cv2

cap = cv2.VideoCapture(0)  # Open webcam
ret, frame1 = cap.read()  # First frame

while True:
    ret, frame2 = cap.read()  # Next frame
    if not ret:
        break

    # Calculate absolute difference
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Display
    cv2.imshow('Motion', thresh)

    # Update frame1 for next iteration
    frame1 = frame2.copy()

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




"""
cam = cv2.VideoCapture(0)
ret, prevFrame = cam.read()
while True:
    ret, currFrame = cam.read()
    diff = cv2.absdiff(prevFrame, currFrame)
    cv2.imshow("Motion", diff)
    x = cv2.waitKey(20)
    c = chr(x & 0xFF)
    if c == "q":
        break
    prevFrame = currFrame
cam.release()
cv2.destroyAllWindows()
"""