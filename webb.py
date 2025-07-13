import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
class_names = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
emojis = {
    'surprise': '😲', 'fear': '😨', 'disgust': '🤢', 'happy': '😄',
    'sad': '😢', 'angry': '😠', 'neutral': '😐'
}
model_path = "D:/AI_real-time emotion detection/best_model_cnn.h5"
input_size = (100, 100)

# --- UI SETUP ---
st.set_page_config(page_title="Real-Time Emotion Detector", layout="wide")
st.title("🧠 Real-Time Facial Emotion Recognition with CNN")
st.write("Upload an image or use webcam for real-time emotion detection")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- IMAGE PREPROCESSING ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, input_size)
    norm = resized / 255.0
    rgb = np.stack([norm] * 3, axis=-1)
    return np.expand_dims(rgb, axis=0)

# --- FACE DETECTION ---
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    return faces

# --- INPUT OPTIONS ---
option = st.radio("Select Input Mode:", ['Upload Image', 'Webcam Real-time'])

# --- UPLOAD IMAGE MODE ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
        faces = detect_face(img)
        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            st.markdown(f"### 🖼️ Detected {len(faces)} face(s)")
            for idx, (x, y, w, h) in enumerate(faces):
                face_crop = img[y:y + h, x:x + w]
                processed = preprocess_image(face_crop)
                pred = model.predict(processed)[0]
                label = class_names[np.argmax(pred)]
                confidence = pred[np.argmax(pred)] * 100

                # Draw bounding box and label on main image
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{emojis[label]} {label} ({confidence:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Display face crop + prediction
                st.markdown(f"**Face {idx+1}: {emojis[label]} {label} ({confidence:.1f}%)**")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(face_crop, caption="Face Region", width=150)
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ax.bar(class_names, pred * 100, color='skyblue')
                    ax.set_ylim([0, 100])
                    ax.set_ylabel('Probability (%)')
                    for i, v in enumerate(pred * 100):
                        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
                    st.pyplot(fig)

            # Show full image with all bounding boxes
            st.markdown("### Full Image with Bounding Boxes")
            resized_img = cv2.resize(img, (450, int(img.shape[0] * 450 / img.shape[1])))
            st.image(resized_img, caption="Annotated Image")

# --- WEBCAM MODE ---
elif option == "Webcam Real-time":
    stframe = st.empty()
    run_btn = st.button("▶ Start Webcam", type="primary")
    stop_placeholder = st.empty()

    if run_btn:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect_face(frame_rgb)
            display_frame = frame.copy()

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_crop = frame_rgb[y:y + h, x:x + w]
                    processed = preprocess_image(face_crop)
                    pred = model.predict(processed)[0]
                    label = class_names[np.argmax(pred)]
                    confidence = pred[np.argmax(pred)] * 100

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{emojis[label]} {label} ({confidence:.1f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            stframe.image(display_frame, channels="BGR")

        cap.release()
