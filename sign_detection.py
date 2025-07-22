import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load model and label encoder
model = tf.keras.models.load_model('sign_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
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

            prediction = model.predict(np.array([landmarks]))[0]
            pred_class = le.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)

            cv2.putText(frame, f'{pred_class} ({confidence:.2f})', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

