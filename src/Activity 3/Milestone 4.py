import os
import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt # Used to display images in Colab

def visualizeResults(image, detection_result):
"""Draws bounding boxes and keypoints on a copy of the input image and return it.
Args:
image: The input RGB image.
detection_result: The list of all "Detection" entities to be visualized.
Returns: Image with bounding boxes.
"""
# Copy the original image and make changes to the copy
annotated_image = image.copy()
height, width, _ = image.shape
for detection in detection_result.detections:
    # Draw bounding_box for each face detected
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
    # Draw face keypoints for each face detected
for keypoint in detection.keypoints:
keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
cv2.circle(annotated_image, keypoint_px, 3, CIRCLE_COLOR, -1)
# Draw category label and confidence score as text on bounding box
category = detection.categories[0]
category_name = category.category_name
category_name = '' if category_name is None else category_name
probability = round(category.score, 2)
result_text = category_name + ' (' + str(probability) + ')'
text_location = (MARGIN + bbox.origin_x,
MARGIN + ROW_SIZE + bbox.origin_y)
cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
return annotated_image
def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
"""Converts normalized value pair to pixel coordinates."""
# Checks if the float value is between 0 and 1.
def is_valid_normalized_value(value):
return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
if not is_valid_normalized_value(normalized_x):
normalized_x = max(0.0, min(1.0, normalized_x))
if not is_valid_normalized_value(normalized_y):
normalized_y = max(0.0, min(1.0, normalized_y))
x_px = min(math.floor(normalized_x * image_width), image_width - 1)
y_px = min(math.floor(normalized_y * image_height), image_height - 1)
return x_px, y_px