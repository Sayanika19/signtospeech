import os
import pickle
import mediapipe as mp
import cv2
import math

# Mediapipe configurations
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directory
DATA_DIR = './data/ASL'

# Classes (Mapping both left and right to the same alphabet/number)
BASE_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CLASSES = {f"{label}_Left": label for label in BASE_CLASSES} 
CLASSES.update({f"{label}_Right": label for label in BASE_CLASSES})

# Initialize lists for data and labels
data = []
labels = []

def calculate_wrist_angle(landmarks):
    """Calculate the wrist angle based on three key points: wrist, index, and pinky."""
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate vectors
    v1 = (index_base.x - wrist.x, index_base.y - wrist.y)
    v2 = (pinky_base.x - wrist.x, pinky_base.y - wrist.y)
    
    # Calculate angle between vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0  # Avoid division by zero
    
    angle = math.acos(dot_product / (mag_v1 * mag_v2))
    return math.degrees(angle)

# Loop through specific classes in the dataset
for class_label, mapped_label in CLASSES.items():
    class_dir = os.path.join(DATA_DIR, class_label)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        continue

    # Process each image in the class directory
    for img_path in os.listdir(class_dir):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(class_dir, img_path))
        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        # Convert image to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            results = hands.process(img_rgb)
        except Exception as e:
            print(f"Error during hand detection: {e}")
            continue

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                # Normalize landmarks relative to bounding box
                min_x, max_x = min(x_), max(x_)
                min_y, max_y = min(y_), max(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append((lm.x - min_x) / (max_x - min_x))  # Normalize x
                    data_aux.append((lm.y - min_y) / (max_y - min_y))  # Normalize y
                
                # Calculate wrist angle
                wrist_angle = calculate_wrist_angle(hand_landmarks.landmark)
                data_aux.append(wrist_angle)  # Add wrist angle

                # Append processed data and labels (Mapping left & right to the same label)
                data.append(data_aux)
                labels.append(mapped_label)  # Assigning alphabet/number as label

        print(f"Processed {img_path} for {class_label}. Data length: {len(data)}, Labels length: {len(labels)}")

# Save data to a pickle file
output_path = 'data_asl.pickle'
print(f"Saving dataset to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Dataset saved successfully.")
