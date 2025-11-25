
import streamlit as st
import av
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode,VideoProcessorBase

# --------------------------------------------
# Load FACE DETECTOR
# --------------------------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# --------------------------------------------
# Load TFLite Emotion Model
# --------------------------------------------
interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


def predict_emotion(face):

    # face shape -> (1, 64, 64, 1)
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return pred


# --------------------------------------------
# Video Transformer Class
# --------------------------------------------
# class EmotionDetector(VideoTransformerBase):
class EmotionProcessor(VideoProcessorBase):

    # def transform(self, frame):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))

            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)       # (1, 64, 64)
            face = np.expand_dims(face, axis=-1)      # (1, 64, 64, 1)

            prediction = predict_emotion(face)
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw box & emotion text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                img,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # return img
        return av.VideoFrame.from_ndarray(img, format="bgr24")



# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------


# webrtc_streamer(
#     key="emotion-detector",
#     mode=WebRtcMode.SENDRECV,
#     # video_transformer_factory=EmotionDetector,
#     video_processor_factory=EmotionProcessor,
#     media_stream_constraints={"video": True, "audio": False},
# )


# webrtc_streamer(
#     key="emotion-detect",
#     mode=WebRtcMode.SENDRECV,
#     video_processor_factory=EmotionProcessor,
#     rtc_configuration={
#         "iceServers": [
#             {"urls": ["stun:stun.l.google.com:19302"]},
#         ]
#     },
#     media_stream_constraints={"video": True, "audio": False},
# )






# ---------------------------
# PAGE LAYOUT WITH TABS
# ---------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("🎭 Face Emotion Detection")


tab1, tab2, tab3, tab4, = st.tabs([
    "📡 Emotion Detection", 
    "📘 About Project", 
    "⚙ How It Works", 
    "🤖 ML & Real Applications"
])



# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Live Emotion Detection")
    st.write("Turn on the camera and the system will detect your emotions in real-time.")
    st.write("👇 Click **Start** to activate your camera & detect emotions in real-time.")

    
   
    webrtc_streamer(
        key="emotion-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
    )


# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("About This Project")
    st.markdown("""
This project shows how a computer can understand human emotions by looking at a face.  
Just like we can look at someone and guess if they are happy or sad, a machine can also learn to do the same.

The project uses Machine Learning to detect emotions such as:

- 😊 Happy  
- 😞 Sad  
- 😡 Angry  
- 😐 Neutral  
- 😮 Surprise  

The goal is to show that even simple school-level projects can use modern MI (Machine Learning) technology easily.
    """)

    st.divider()

    st.subheader("Team")
    st.markdown("""
**👧 Project Made By:**  
**Sara (Class 9)**  
DR.S. Radha Krishnan Public Inter College, Jhansi  

**👦 Guided By:**  
**Shan Khan (Brother)**  
    """)



# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("How The Project Works")
    st.markdown("""
### 🔄 Step-by-Step Flow

1. **📷 Camera Captures the Face**  
   The webcam takes a live video.

2. **👤 Face Detection**  
   The system finds where the face is in the image.

3. **🖼 Image Preprocessing**  
   The face is resized into a 64×64 grayscale image.

4. **🧠 Emotion Prediction**  
   The ML model predicts the emotion based on patterns it learned.

5. **🖥 Output**  
   The emotion is shown on the screen with a box around the face.

This happens in **real-time** using Streamlit + WebRTC.
    """)






with tab4:
    st.subheader("🤖 What is Machine Learning?")
    st.markdown("""
Machine Learning is a technology that allows a computer to **learn from examples** instead of being directly programmed.

### 🧠 Simple Explanation
- If we show the computer many pictures of **happy faces**,  
- And many pictures of **sad or angry faces**,  
- The computer learns the **patterns** from these images.

Later, when a **new face** is shown, it tries to guess the emotion by comparing it with what it has learned earlier.

Machine Learning works just like how children learn:
**By looking, understanding, practicing, and remembering.**

---

## 🌍 Real-World Applications of Machine Learning

Machine Learning is used in almost every industry today.  
Here are some easy-to-understand examples:

### 🏥 1. Healthcare
- Detecting diseases from X-ray and MRI images  
- Monitoring patient emotions and stress  
- Predicting heart attacks  

### 📱 2. Mobile Phones
- Face Unlock  
- Voice assistants (Siri, Google Assistant)  
- Automatically grouping photos  

### 🛒 3. E-Commerce
- Amazon product recommendations  
- Detecting fake reviews  
- Price prediction  

### 🚗 4. Transportation
- Self-driving cars  
- Traffic prediction  
- Auto lane detection  

### 🎥 5. CCTV & Security
- Crowd emotion monitoring  
- Detecting suspicious activity  
- Face recognition  

### 🏫 6. Education
- Smart attendance using face detection  
- Detecting if students look confused  
- Personalized learning systems  

### 💼 7. Business & Banking
- Fraud detection  
- Customer support chatbots  
- Credit score prediction  

---

### 🎯 Why It Matters?
Machine Learning is one of the most important technologies of the future.  
It helps computers:
- Make decisions  
- Recognize images  
- Understand voices  
- Predict events  

And this project is a **simple example** of how ML can detect human emotions.
    """)
