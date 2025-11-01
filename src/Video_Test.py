from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import math

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green



def visualizeResults(rgb_image, detection_result):
    """
    Draws hand skeleton for each hand visible in an image
    :param rgb_image: An RGB image array
    :param detection_result: The results from the hand landmark detector
    :return: a copy of the input array with the hand skeletons drawn on it
    """
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
        hand_landmarks])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


if __name__ == "__main__":

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Set up model options, and load trained model

    """
    modelPath = "???"  # TODO: Put correct path and model name here
    base_options = python.BaseOptions(model_asset_path=modelPath)  # All models use the same base options
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    """

    # TODO: Replace None below with the correct call to create the model


    # Set up camera
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        failCount = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                failCount += 1
                if failCount < 5:
                    continue
                else:
                    break
            failCount = 0

            """
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detect_result = detector.detect(mp_image)
            """


            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow("Detected", image)

            x = cv2.waitKey(30)
            ch = chr(x & 0xFF)
            if ch == 'q':
                break


        cap.release()

        cv2.destroyAllWindows()

