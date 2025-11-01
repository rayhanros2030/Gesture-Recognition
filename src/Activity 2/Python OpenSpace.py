import cv2
import numpy as np

image = cv2.imread("../SampleImages/coins5.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if image is None:
    print("Error: Could not load image.")
    exit()

# Apply Triangle thresholding
# Returns the threshold value and the binary image
thresh_value, triangle_image = cv2.threshold(image, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# Save and display results
cv2.imwrite("../EXTRA FILES/triangle_output.jpg", triangle_image)
cv2.imshow("Original Grayscale", image)
cv2.imshow("Triangle Threshold", triangle_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

