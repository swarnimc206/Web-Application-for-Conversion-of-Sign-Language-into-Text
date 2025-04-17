import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from PIL import Image

# Load the trained model
model = load_model("sign_language_model.h5")

# Class labels (Ensure this matches your dataset's class indices)
class_labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
    22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S',
    29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((64, 64))  # Resize image to 64x64
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App
st.title("Sign Language Recognition")
st.text("Upload an image or use your webcam to predict the corresponding letter/number.")

# Option for image upload or webcam
mode = st.radio("Choose input method:", ("Upload an Image", "Capture from Webcam"))

# Webcam option
if mode == "Capture from Webcam":
    st.text("Use your webcam to capture a sign language gesture.")

    # Custom video transformer for webcam processing
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convert frame to BGR format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(img_rgb)  # Convert to PIL Image

            # Preprocess and predict
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]

            # Add prediction text to the video frame
            cv2.putText(
                img, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
            return img

    # Start the webcam stream
    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

# Image upload option
elif mode == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        
        # Display prediction
        st.write(f"Predicted Label: {predicted_label}")
