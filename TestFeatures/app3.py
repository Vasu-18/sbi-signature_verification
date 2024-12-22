import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from skimage import io
from skimage.filters import threshold_otsu
from scipy import ndimage

# Load your model (update with your model's path)
@st.cache_resource  # Cache the loaded model for performance
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\garvi\Downloads\SBI_Life1\SBI_Life1\SBILife1.py")
    return model

# Function to preprocess the image for your model
def preprocess_image(image_path):
    img = io.imread(image_path)
    # Update with your model's required preprocessing steps
    grey = np.mean(img, axis=2)  # Example: Convert to grayscale
    binary_img = grey > threshold_otsu(grey)
    processed_img = ndimage.binary_fill_holes(binary_img).astype(np.float32)
    processed_img = np.expand_dims(processed_img, axis=(0, -1))  # Add batch and channel dimensions
    return processed_img

# Function to verify signature using the model
def verify_signature(image_path, model):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)  # Get prediction
    # Assuming a binary classification output (0 for genuine, 1 for forged)
    result = "Genuine" if np.argmax(prediction) == 0 else "Forged"
    return result

# Streamlit UI
st.title("NITI SURAKSHA - Signature Verification")
st.write("Welcome to **NITI SURAKSHA**. Upload a signature image below to verify if it is genuine or forged.")

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.subheader("Uploaded Signature:")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Signature", use_column_width=True)

    # Run verification
    if st.button("Verify Signature"):
        result = verify_signature(temp_path, model)
        st.subheader("Verification Result:")
        if result == "Genuine":
            st.success("The uploaded signature is **Genuine**.")
        else:
            st.error("The uploaded signature is **Forged**.")
    
    # Clean up temporary files
    os.remove(temp_path)
