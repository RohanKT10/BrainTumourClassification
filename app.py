import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('final_model.keras')

# Class labels
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# Function to preprocess the uploaded image

def preprocess_image(image_input):
    image_input = image_input.resize((224, 224))  # Resize to match model input
    image_input = np.array(image_input)  # Convert to numpy array
    image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)  # Ensure correct color channel order
    image_input = np.array([image_input])
    return image_input


# Display prediction results as a bar graph
def show_confidence_chart(prediction, class_names):
    fig, ax = plt.subplots()
    bars = ax.barh(class_names, prediction, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722'])
    ax.set_xlim(0, 1)
    ax.set_title("Confidence Scores")

    # Add labels inside bars
    for bar, value in zip(bars, prediction):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value * 100:.2f}%", va='center')

    st.pyplot(fig)


# Streamlit App Layout
st.title("🧠 Brain Tumor Classification")

# About Project Section
st.markdown("###  About this Project")
st.markdown(
    "This Brain Tumor Classification project leverages a deep learning model based on EfficientNetB0 to classify MRI images into four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor. The model was trained using augmented data for improved robustness and generalization. With an intuitive Streamlit interface, users can upload MRI images for instant analysis and receive clear predictions along with confidence scores.Simply upload an MRI image to see the prediction.")

st.markdown("### 📌 Instructions")
st.markdown(
    "1. Upload an MRI image in JPG, PNG, or BMP format.\n2. Wait for the model to analyze the image.\n3. View the predicted class and confidence scores.")

# File uploader with option to remove file
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg", "bmp", "tiff"], key="uploader")

if uploaded_file is not None:
    try:
        with st.spinner("🔎 Analyzing Image... Please wait!"):
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Uploaded Image", width=450)

            # Preprocess and predict
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0]  # Extract 1D array

            # Display results
            st.subheader("📊 Prediction Results:")
            predicted_class = class_names[np.argmax(prediction)]
            st.success(f"**Predicted Class:** {predicted_class}")

            show_confidence_chart(prediction, class_names)
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by **Rohan KT** | Brain Tumor Classification with EfficientNetB0")