import cv2
import mediapipe as mp
import time
import numpy as np
from math import sqrt

# Constants for landmarks
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20

# Initialize the MediaPipe solutions
handSolution = mp.solutions.hands
hands = handSolution.Hands()
faceSolution = mp.solutions.face_mesh
face_mesh = faceSolution.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils  # Drawing utilities

# Opening camera (0 for the default camera)
videoCap = cv2.VideoCapture(0)
lastFrameTime = 0
frame = 0

while True:
    frame += 1
    # Reading image
    success, img = videoCap.read()
    if success:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # FPS calculations
        thisFrameTime = time.time()
        fps = 1 / (thisFrameTime - lastFrameTime)
        lastFrameTime = thisFrameTime

        # Write FPS on the image
        cv2.putText(img, f'FPS: {int(fps)}',
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process face landmarks
        recFaces = face_mesh.process(imgRGB)
        if recFaces.multi_face_landmarks:
            for face_landmarks in recFaces.multi_face_landmarks:
                mp_draw.draw_landmarks(img, face_landmarks, faceSolution.FACEMESH_CONTOURS, 
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), 
                                       mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))

        # Recognize hands from the image
        recHands = hands.process(imgRGB)
        if recHands.multi_hand_landmarks:
            for i, hand in enumerate(recHands.multi_hand_landmarks):
                # Draw the dots and connections on the image
                mp_draw.draw_landmarks(img, hand, handSolution.HAND_CONNECTIONS)

                # Add label for hand (left or right)
                if recHands.multi_handedness:
                    handedness = recHands.multi_handedness[i]
                    hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                    score = handedness.classification[0].score  # Confidence score
                    h, w, c = img.shape
                    cx, cy = int(hand.landmark[0].x * w), int(hand.landmark[0].y * h)
                    cv2.putText(img, f'{hand_label} ({score:.2f})',
                                (cx - 50, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Get the positions of thumb tip and all finger tips
                h, w, c = img.shape
                thumb_tip = hand.landmark[THUMB_TIP]
                finger_tips = {
                    "Index": hand.landmark[INDEX_FINGER_TIP],
                    "Middle": hand.landmark[MIDDLE_FINGER_TIP],
                    "Ring": hand.landmark[RING_FINGER_TIP],
                    "Pinky": hand.landmark[PINKY_TIP]
                }

                # Convert thumb tip to pixel coordinates
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)

                # Calculate distances for each finger tip and display them
                distances = []
                for finger_name, finger_tip in finger_tips.items():
                    finger_x, finger_y = int(finger_tip.x * w), int(finger_tip.y * h)
                    # Calculate Euclidean distance
                    distance = sqrt((thumb_x - finger_x) ** 2 + (thumb_y - finger_y) ** 2)
                    # Highlight fingertip
                    cv2.circle(img, (finger_x, finger_y), 10, (0, 255, 0), cv2.FILLED)
                    # Display distance near fingertip
                    cv2.putText(img, f'{finger_name}: {int(distance)} px',
                                (finger_x + 10, finger_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # Add to the list of distances
                    distances.append(distance)

        # Display the image
        cv2.imshow("CamOutput", img)
        cv2.waitKey(1)

# Release the camera and close all OpenCV windows
videoCap.release()
cv2.destroyAllWindows()
