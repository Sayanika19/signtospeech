import os
import cv2
import mediapipe as mp

# Configuration for Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Load the dataset
data_path = './data/ASL'
classes = os.listdir(data_path)

# Directory to save images with landmarks
output_path = './landmarked_images'
os.makedirs(output_path, exist_ok=True)

# Iterate over each class in the dataset
for class_label in classes:
    class_dir = os.path.join(data_path, class_label)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {class_label}")

    # Flags to save one image for each hand type
    left_hand_saved = False
    right_hand_saved = False

    # Iterate over each image in the class directory
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hands
        results = hands.process(image_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label  # "Left" or "Right"

                # Draw the landmarks on the image
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Save the image for the corresponding hand type
                if hand_label == "Left" and not left_hand_saved:
                    output_file = os.path.join(output_path, f"{class_label}_Left.jpg")
                    cv2.imwrite(output_file, annotated_image)
                    print(f"Saved left hand image for class {class_label} at {output_file}")
                    left_hand_saved = True

                elif hand_label == "Right" and not right_hand_saved:
                    output_file = os.path.join(output_path, f"{class_label}_Right.jpg")
                    cv2.imwrite(output_file, annotated_image)
                    print(f"Saved right hand image for class {class_label} at {output_file}")
                    right_hand_saved = True

        # Break after saving one image per hand type for the class
        if left_hand_saved and right_hand_saved:
            break

# Release resources
hands.close()
print("Landmark images saved for all classes with left and right hands.")
