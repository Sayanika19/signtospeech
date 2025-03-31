import os
import cv2
import time
# Constants
DATA_DIR = './data'
SIGN_LANGUAGES = ["ASL"]  # American Sign Language
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]  # Added J and Z
NUMBERS = [str(i) for i in range(10)]  # Numbers 0-9
CLASSES = [f"{label}_Left" for label in ALPHABETS + NUMBERS] + [f"{label}_Right" for label in ALPHABETS + NUMBERS]  # Include left and right variations
DATASET_SIZE = 100  # Number of images per class

# Create directories
for language in SIGN_LANGUAGES:
    language_dir = os.path.join(DATA_DIR, language)
    os.makedirs(language_dir, exist_ok=True)
    for label in CLASSES:
        label_dir = os.path.join(language_dir, label)
        os.makedirs(label_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

def capture_data(language, label):
    """
    Captures frames for a specific sign.
    Special handling for J and Z to track motion.
    """
    class_dir = os.path.join(DATA_DIR, language, label)
    print(f'Preparing to collect data for {language} - {label}')

    counter = 0
    recording = False  # Flag to start recording
    
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame
        
        # Display frame with instructions
        cv2.putText(frame, f'{language} - {label}: Frame {counter+1}/{DATASET_SIZE}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, 'Press R to start recording, S to skip, Q to quit',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('frame', cv2.resize(frame, (640, 480)))

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Quit
            print("Recording stopped by user.")
            break
        elif key == ord('s'):  # Skip
            print(f"Skipping data collection for {language} - {label}.")
            return
        elif key == ord('r'):  # Start recording
            recording = not recording
            print(f"{'Started' if recording else 'Stopped'} recording for {language} - {label}.")

        if recording:
            filename = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(filename, frame)
            counter += 1
            print(f"Captured {counter} image(s) for {language} - {label}.")
            
            # Special handling for J and Z (motion tracking)
            if "J" in label or "Z" in label:
                time.sleep(0.1)  # Capture rapid sequential frames

    print(f'Finished collecting data for {language} - {label}.')

# Start data collection
for language in SIGN_LANGUAGES:
    for label in CLASSES:
        capture_data(language, label)

# Release resources
cap.release()
cv2.destroyAllWindows()
