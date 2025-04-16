import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
from PIL import Image
import av

# Set page config
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="✋",
    layout="wide"
)

# Define class labels directly (update this to match your model's classes)
CLASS_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
}

# Load model
@st.cache_resource
def load_model_cached():
    try:
        return load_model("sign_language_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model_cached()

# Preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Webcam processor
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
    
    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        if self.frame_count % 5 != 0:
            return img
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_array = preprocess_image(pil_img)
        
        try:
            predictions = model.predict(img_array, verbose=0)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions[0])
            
            if confidence > 0.7:
                label = CLASS_LABELS.get(predicted_class, "Unknown")
                cv2.putText(
                    img, f"{label} ({confidence:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                )
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return img

# Rest of your app code remains the same...
# [Keep all the Streamlit UI code from the previous version]
