import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model and label encoder
with open('model_asl_rf.p', 'rb') as model_file:
    model_data = pickle.load(model_file)
    model = model_data['model']
    label_encoder = model_data['label_encoder']

# Mediapipe configurations
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Helper function to preprocess hand landmarks
def preprocess_landmarks(landmarks):
    x_ = [lm.x for lm in landmarks]
    y_ = [lm.y for lm in landmarks]

    min_x, max_x = min(x_), max(x_)
    min_y, max_y = min(y_), max(y_)

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append((lm.x - min_x) / (max_x - min_x))  # Normalize x
        normalized_landmarks.append((lm.y - min_y) / (max_y - min_y))  # Normalize y

    # Calculate wrist angle
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    v1 = (index_base.x - wrist.x, index_base.y - wrist.y)
    v2 = (pinky_base.x - wrist.x, pinky_base.y - wrist.y)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)

    wrist_angle = 0.0
    if mag_v1 != 0 and mag_v2 != 0:
        wrist_angle = np.degrees(np.arccos(dot_product / (mag_v1 * mag_v2)))

    normalized_landmarks.append(wrist_angle)

    return normalized_landmarks

# Video capture and prediction
def classify_gestures():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Preprocess landmarks
                data_point = preprocess_landmarks(hand_landmarks.landmark)

                # Make prediction
                prediction = model.predict([data_point])
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                # Calculate bounding box from landmarks
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * frame.shape[1])
                y_min = int(min(y_coords) * frame.shape[0])
                x_max = int(max(x_coords) * frame.shape[1])
                y_max = int(max(y_coords) * frame.shape[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks on the hand
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Show the frame
        cv2.imshow('Gesture Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_gestures()


def get_prediction(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction_text = "  "

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_point = preprocess_landmarks(hand_landmarks.landmark)

            try:
                prediction = model.predict([data_point])
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                prediction_text = predicted_label
            except:
                prediction_text = "Error in prediction"

            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame, prediction_text
