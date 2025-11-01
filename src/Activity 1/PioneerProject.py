import cv2
img = cv2.imread("SampleImages/wildColumbine.jpg")
cv2.imshow("Original", img)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res, img3 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

# Load an image from your local directory # Replace with your image path
image = cv2.imread("SampleImages/snowLeo1.jpg")

# Check if image loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# 1. Pencil Sketch Effect
# pencilSketch() returns two outputs: a grayscale sketch and a color sketch
gray_sketch, color_sketch = cv2.pencilSketch(image, sigma_s=80, sigma_r=0.1, shade_factor=0.1)

# Save the pencil sketch outputs
cv2.imwrite("pencil_sketch_gray.jpg", gray_sketch)
cv2.imwrite("pencil_sketch_color.jpg", color_sketch)

# 2. Stylization Effect
# stylization() creates a watercolor-like effect
stylized_image = cv2.stylization(image, sigma_s=90, sigma_r=0.7)

# Save the stylized output
cv2.imwrite("stylized_image.jpg", stylized_image)

# Optional: Display the results (remove or comment out if you don't want windows popping up)
cv2.imshow("Original Image", image)
cv2.imshow("Pencil Sketch (Gray)", gray_sketch)
cv2.imshow("Pencil Sketch (Color)", color_sketch)
cv2.imshow("Stylized Image", stylized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

-------------------------------------------------------------------

draw1 = np.zeros((500, 500, 3), np.uint8)

cv2.setMouseCallback('Image Mouse', draw_circle)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(draw1, "Hi, there", (10, 270), font, 1, (255, 255, 255))
cv2.imshow("Black", draw1)
cv2.imwrite("blackPic.jpg", draw1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("SampleImages/snowLeo2.jpg")
cv2.imshow("Original", img)
(rows, cols, depth) = img.shape
rotMat = cv2.getRotationMatrix2D( (cols / 2, rows / 2), 45, 1)
rotImg = cv2.warpAffine(img, rotMat, (cols, rows))
cv2.imshow("Rotated", rotImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("SampleImages/snowLeo2.jpg")
(rows, cols, dep) = img.shape
cv2.imshow("Original", img)
transMatrix = np.float32([[1, 0, 30], [0, 1, 50]]) # change 30 and 50
transImag = cv2.warpAffine(img, transMatrix, (cols, rows))
cv2.imshow("Translated", transImag)
cv2.waitKey(0)
cv2.destroyAllWindows()


vidCap = cv2.VideoCapture(0)
for i in range(300):
    ret, img = vidCap.read()
    cv2.imshow("Webcam", img)
    cv2.waitKey(10)
cv2.destroyAllWindows()
vidCap.release()



draw1 = np.zeros((300, 500, 3), np.uint8)
draw2 = 255 * np.ones((500, 300, 3), np.uint8)
cv2.line(draw2, (50, 50), (150, 250), (0, 0, 255))
cv2.rectangle(draw1, (10, 100), (100, 10), (0, 180, 0), -1)
cv2.circle(draw2, (30, 30), 30, (220, 0, 0), -1)
cv2.ellipse(draw1, (250, 150), (100, 60), 30, 0, 220, (250, 180, 110), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(draw1, "Hi, there", (10, 270), font, 1, (255, 255, 255))
cv2.imshow("Black", draw1)
cv2.imshow("White", draw2)
cv2.imwrite("blackPic.jpg", draw1)
cv2.imwrite("whitePic.jpg", draw2)
cv2.waitKey(0)
cv2.destroyAllWindows()


catImage = cv2.imread("SampleImages/snowLeo1.jpg")
faceROI = catImage[250:550, 570:860, :]
cv2.imshow("Orig", catImage)
cv2.imshow("Face", faceROI)
cv2.waitKey(0)
# set blue channel of this ROI to zero, notice change shows in original
faceROI[:, :, 1] = 0
cv2.imshow("Orig", catImage)
cv2.imshow("Face", faceROI)
cv2.waitKey(0)
# flip the face upside down by reversing the X direction and keeping the others the same
flipFace = faceROI[::-1, :, :]
cv2.imshow("Flipped", flipFace)
cv2.waitKey(0)


image = cv2.imread("SampleImages/canyonlands.jpg")
print("Value at row 20, column 20:", image[20,20], image[20, 20, :])
print("Row 5:")
print(image[5, :, :])
print("Column 0:")
print(image[:, 0, :])
print("Small section:")
print(image[20:60, 100:200, :])
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


origImage = cv2.imread("SampleImages/snowLeo4.jpg")
gray = cv2.cvtColor(origImage, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", gray)
blankImg1 = np.zeros((400, 250), np.uint8) #np.uint8 means an 8-bit integer type
cv2.imshow("Black background image", blankImg1)
blankImg2 = 255 * np.ones((300, 300), np.uint8)
cv2.imshow("White background image", blankImg2)
cv2.waitKey(0)
cv2.destroyAllWindows()




image = cv2.imread("SampleImages/antiqueTractors.jpg")
(bc, gc, rc) = cv2.split(image)
# each channel is shown as grayscale, because it only has value per pixel
cv2.imshow("Blue channel", bc)
cv2.imshow("Green channel", gc)
cv2.imshow("Red channel", rc)
cv2.moveWindow("Blue channel", 30, 30)
cv2.moveWindow("Green channel", 330, 60)
cv2.moveWindow("Red channel", 630, 90)
cv2.waitKey(0)
# Put image back together again
imCopy = cv2.merge((bc, gc, rc))
cv2.imshow("Image Copy", imCopy)
cv2.waitKey(0)


img1 = cv2.imread("SampleImages/snowLeo1.jpg")
cv2.imshow("Leopard 1", img1) # This is basically a way of printing out images
img2 = cv2.imread("SampleImages/snowLeo2.jpg") # putting a picture into an array
cv2.imshow("Leopard 2", img2)
cv2.waitKey(0) # tells the program to wait indefinitely until a user presses a key
cv2.destroyAllWindows() # until the key has been pressed, the images will remain
"""

