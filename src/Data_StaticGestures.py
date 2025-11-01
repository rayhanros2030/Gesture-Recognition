import cv2
import os
from datetime import datetime

# 🗂️ Set your target folder to save screenshots
save_folder = "Thumbsup"
os.makedirs(save_folder, exist_ok=True)

# 📸 Start webcam
cap = cv2.VideoCapture(0)
print("Press 's' to take a screenshot. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip to mirror the image (optional)
    frame = cv2.flip(frame, 1)
    cv2.imshow("Live Feed - Press 's' to Screenshot", frame)

    key = cv2.waitKey(1) & 0xFF

    # 📸 Press 's' to take screenshot
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved to: {filename}")

    # ❌ ESC key to quit
    elif key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()