# Real-Time Sign Language Detection System

## Overview

This project focuses on building a **real-time Sign Language Detection System** using **computer vision and machine learning techniques**. The system accurately recognizes and interprets **American Sign Language (ASL) gestures**, helping bridge the communication gap between sign language users and non-users.

Recent updates have improved **motion tracking** and **gesture recognition**, reducing ambiguity in dynamic gestures like **J & Z**, as well as **number detection (0-9)**. The system now also supports **Sign-to-Text and Sign-to-Speech in multiple languages**, making communication more inclusive and accessible.

## Technologies Used

- **OpenCV**: Image processing and video stream handling
- **MediaPipe Hands**: Hand landmark detection and feature extraction
- **scikit-learn (Random Forest Classifier)**: Machine learning model for classification
- **Flask**: Web framework for hosting a local site to showcase the project
- **Google Translate API**: Enables multilingual support for text and speech translation
- **pyttsx3**: Converts recognized gestures into speech output
- **Python**: The core programming language for implementation

## Key Features & Improvements

- **Real-Time Gesture Recognition** – Accurately detects ASL gestures using webcam input.
- **J & Z Gesture Recognition** – Improved motion tracking resolves ambiguity in dynamic gestures.
- **Numbers 0-9 Detection** – Expands recognition to numerical signs.
- **Sign-to-Text & Sign-to-Speech** – Converts recognized gestures into written and spoken words.
- **Multilingual Support** – Users can select their preferred language for text and speech output.
- **Web Interface with Flask** – Live demonstration via a local site for real-time interaction.

## System Workflow

1. **Data Collection**: ASL gesture images are captured and labeled.
2. **Preprocessing**: Extracts hand landmarks, normalizes data, and computes additional features like wrist angles.
3. **Model Training**: The **Random Forest Classifier** is trained on preprocessed data.
4. **Real-Time Prediction**: Webcam frames are processed, and the system predicts gestures live.
5. **Speech & Translation**: The detected gesture is converted into text and spoken aloud in the selected language.
6. **Web-based Interface**: Users can interact with the system via a local website powered by Flask.

## How It Works

1. **Run the Flask server** to start the local web app.
2. **Access the web interface** to see real-time gesture recognition.
3. **Change language settings** for multilingual speech output.

## Installation & Usage

### Prerequisites

Ensure you have Python installed and install the required dependencies:

```
pip install opencv-python mediapipe scikit-learn flask pyttsx3 googletrans==4.0.0-rc1

```

### Running the Application

1. Clone the repository:
    
    ```
    git clone https://github.com/pratyaksha0612/ASL-Detection.git
    
    ```
    
2. Navigate to the project directory:
    
    ```
    cd ASL-Detection
    
    ```
    
3. Run the Flask application:
    
    ```
    python app.py
    
    ```
    
4. Open your browser and go to:
    
    ```
    http://127.0.0.1:5000/
    
    ```
    

## Future Improvements

To further enhance our **Sign Language Detection System**, we aim to introduce:

- **Expand Gesture Library** – We plan to add more ASL signs, including complex two-handed gestures and contextual expressions, making the system more versatile.

- **Improve Accuracy with Deep Learning** – Implementing CNNs or Transformer-based models will enhance recognition accuracy, especially for similar-looking gestures and dynamic signs.
- **Mobile & Edge AI Support** – Developing a lightweight version optimized for **mobile devices** using **TensorFlow Lite** or **MediaPipe Edge AI** will allow real-time detection on smartphones without relying on high-performance computers.
- **Offline Language Translation** – Currently, the translation feature relies on internet-based APIs. We aim to integrate an **offline translation model** so that users can convert sign language to text/speech without needing an internet connection.
- **Enhanced UI & Accessibility Features** – We plan to refine the web interface with **better user experience, voice commands, and accessibility options**, ensuring inclusivity for all users.

## **👥 Contributors & Acknowledgments**

This project is developed and maintained by:

- **Pratyaksha Singh** 🛠️ | *Machine Learning & Computer Vision*
- **Priyam Kundu** 💡 | *System Development & Web Integration*

A big thank you to the **open-source community** and the developers of **MediaPipe, OpenCV, scikit-learn, Flask, and Google Translate API**, whose tools made this project possible!



![2](https://github.com/user-attachments/assets/8c667b62-ebd1-4db6-a6d2-5bcf96357101)



![A_Right_Right](https://github.com/user-attachments/assets/56569088-d515-4181-83d4-1ffb2465e9a8)



![Screenshot 2024-12-17 223620](https://github.com/user-attachments/assets/73a9d74c-e78f-4cb0-abb0-8078c46755c2)
