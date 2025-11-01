import cv2

image = cv2.VideoCapture(0)

# Check if video opened successfully

# Process video frame by frame
while True:
    # Read a frame
    ret, frame = image.read()

    # Break loop if no frame is returned (end of video)
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 25, 50)

    # Display the original frame and edges
    cv2.imshow('Original Video', frame)
    cv2.imshow('Canny Edges', edges)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


