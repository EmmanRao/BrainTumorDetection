import pickle
import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import os

# Load the trained model
@st.cache_resource  # Cache the model for faster subsequent loads
def load_model():
    try:
        # Load model structure and weights
        with open('models/CNN_model_structure.json', 'r') as file:
            model_structure = file.read()
        model = tf.keras.models.model_from_json(model_structure)

        with open('models/CNN_weights.pkl', 'rb') as file:
            weights = pickle.load(file)
        model.set_weights(weights)

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the model
model = load_model()

# Helper function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the required dimensions
    if image.mode != 'RGB':  # Ensure the image is in RGB format
        image = image.convert('RGB')
    image_array = np.expand_dims(np.array(image), axis=0)  # Add batch dimension
    return image_array

# Main app
st.title("Brain Tumor Classification App")
st.write(
    """
    Upload an MRI image to classify it as one of the following categories:
    - Glioma
    - Meningioma
    - No Tumor
    - Pituitary Tumor
    """
)

# Image upload section
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make prediction
        image_array = preprocess_image(image)
        if model:
            predictions = model.predict(image_array)
            predicted_label_index = np.argmax(predictions[0])

            # Define class labels
            class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
            predicted_label = class_labels[predicted_label_index]

            # Display the prediction
            st.success(f"Prediction: {predicted_label}")
        else:
            st.error("Model could not be loaded. Please check the backend.")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid JPG or PNG image.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
