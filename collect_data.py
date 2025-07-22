import cv2
import mediapipe as mp
import numpy as np
import os

# Setup
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter label (A-Z): ")
num_samples = 0
SAMPLES_PER_CLASS = 200

while num_samples < SAMPLES_PER_CLASS:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            landmarks = np.array(landmarks)

            file_name = os.path.join(DATA_DIR, f'{label}_{num_samples}.npy')
            np.save(file_name, landmarks)
            num_samples += 1

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
