import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
from PIL import Image
import av
import os

# Set page config
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="✋",
    layout="wide"
)

# Load model and class indices
@st.cache_resource
def load_models():
    try:
        model = load_model("sign_language_model.h5")
        class_indices = np.load("class_indices.npy", allow_pickle=True).item()
        class_labels = {v:k for k,v in class_indices.items()}
        return model, class_labels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, class_labels = load_models()

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
        
        # Only process every 5th frame for performance
        if self.frame_count % 5 != 0:
            return img
        
        # Convert and predict
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_array = preprocess_image(pil_img)
        
        try:
            predictions = model.predict(img_array, verbose=0)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions[0])
            
            if confidence > 0.7:  # Confidence threshold
                label = class_labels.get(predicted_class, "Unknown")
                cv2.putText(
                    img, f"{label} ({confidence:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
                )
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return img

# Main app
st.title("American Sign Language Recognition System")
st.markdown("""
    This application recognizes American Sign Language (ASL) letters and numbers 
    from images or webcam input.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Input Mode:",
        ("Image Upload", "Real-time Webcam"),
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
        **Instructions:**
        - For image upload: Use clear, well-lit images of hand signs
        - For webcam: Position your hand in the center of the frame
        - Only letters A-Z and numbers 0-9 are supported
    """)

# Main content
if model is None or class_labels is None:
    st.error("Model failed to load. Please check if the model files exist.")
    st.stop()

if mode == "Image Upload":
    st.subheader("Upload an ASL Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
        
        with col2:
            with st.spinner("Processing..."):
                try:
                    img_array = preprocess_image(image)
                    predictions = model.predict(img_array, verbose=0)
                    confidence = np.max(predictions)
                    predicted_class = np.argmax(predictions[0])
                    
                    if confidence > 0.7:
                        label = class_labels.get(predicted_class, "Unknown")
                        st.success(f"Prediction: **{label}**")
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Show prediction distribution
                        top_n = 5
                        top_indices = np.argsort(predictions[0])[-top_n:][::-1]
                        st.subheader("Top Predictions:")
                        for i in top_indices:
                            st.progress(float(predictions[0][i]), 
                            st.caption(f"{class_labels.get(i, 'Unknown')}: {predictions[0][i]:.2%}")
                    else:
                        st.warning("The model is not confident about this prediction.")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

else:  # Webcam mode
    st.subheader("Real-time ASL Recognition")
    st.warning("Webcam feature requires camera access permission.")
    
    webrtc_ctx = webrtc_streamer(
        key="asl-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if not webrtc_ctx.state.playing:
        st.info("Waiting for webcam to start...")
        st.image("https://via.placeholder.com/640x360?text=Webcam+Feed+Will+Appear+Here")

# Footer
st.markdown("---")
st.markdown("""
    *Note: This is a demonstration system. Accuracy may vary based on lighting, 
    hand position, and background conditions.*
""")
