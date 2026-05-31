import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load model and labels
model = load_model('model/sign_language_model.h5')

with open('model/class_labels.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the dictionary: {0: 'A', 1: 'B', ...}
labels = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = 64

print("Starting prediction... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_label = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # Get hand bounding box
        h, w, _ = frame.shape
        x_list = [lm.x for lm in result.multi_hand_landmarks[0].landmark]
        y_list = [lm.y for lm in result.multi_hand_landmarks[0].landmark]

        x_min = max(0, int(min(x_list) * w) - 30)
        y_min = max(0, int(min(y_list) * h) - 30)
        x_max = min(w, int(max(x_list) * w) + 30)
        y_max = min(h, int(max(y_list) * h) + 30)

        # Crop and preprocess hand
        hand_crop = frame[y_min:y_max, x_min:x_max]

        if hand_crop.size > 0:
            hand_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            hand_crop = hand_crop / 255.0
            hand_crop = np.expand_dims(hand_crop, axis=0)

            # Predict
            prediction = model.predict(hand_crop, verbose=0)
            confidence = np.max(prediction)
            class_id = np.argmax(prediction)
            predicted_label = labels[class_id]

            # Only show if confidence is high enough
            if confidence < 0.5:
                predicted_label = "..."

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display predicted letter
    cv2.putText(frame, f"Sign: {predicted_label}",
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
               1.5, (0, 255, 0), 3)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()