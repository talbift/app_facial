import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your model
model = load_model("emotion_model.h5")  # <-- replace with your actual path
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Setup Mediapipe
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

def detect_and_crop_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            face = image[y:y + h, x:x + w]
            mp_draw.draw_detection(image, detection)
            return face, image
    return None, image

def preprocess_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

def predict_emotion(face):
    processed = preprocess_face(face)
    prediction = model.predict(processed)
    return emotion_labels[np.argmax(prediction)]

# Streamlit UI
st.title("Facial Emotion Recognition App 😄😠😢😲")
st.write("This app uses MediaPipe + CNN to detect facial emotions in real time.")

option = st.radio("Choose input type:", ["Webcam", "Upload Image"])

if option == "Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break

        face, annotated = detect_and_crop_face(frame)

        if face is not None:
            emotion = predict_emotion(face)
            cv2.putText(annotated, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    cap.release()

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        face, annotated = detect_and_crop_face(image)

        if face is not None:
            emotion = predict_emotion(face)
            cv2.putText(annotated, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Predicted: {emotion}")
        else:
            st.warning("No face detected.")
