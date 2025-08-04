import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter

# Load the trained KNN model
knn = joblib.load("knn3_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

frames = 50

currentframe = 0

predictions = []

maxprediction = 0

alphabet = ['a', 'b','c', 'd', 'e', 'f']


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract the 21 hand landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

# Flatten the landmarks into a 1D array
            landmarks = np.array(landmarks).flatten()

            # Predict the letter using the KNN model
            prediction = knn.predict([landmarks])[0]
            predictions.append(prediction)
            # print(prediction, "prediction")
            # Display the prediction on the screen
            cv2.putText(frame, f"Prediction: {alphabet[maxprediction]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("ASL Prediction", frame)
    if(len(predictions) > 20):
        predictionscount = Counter(predictions)
        maxprediction = predictionscount.most_common(1)[0][0]
        predictions = []
        print(predictionscount, alphabet[maxprediction])
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()