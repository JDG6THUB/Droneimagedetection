import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# A function to load the model and cache it.
# This prevents the model from being reloaded every time the user interacts with the app.
@st.cache_resource
def load_model():
    # Load the model from the saved folder
    model = tf.keras.models.load_model('Drone_image_human_detection_model.keras')
    return model

# Load the model once
model = load_model()

# Set up the Streamlit app title and description
st.title("Drone Image Person Detector")
st.write("Upload an image to check if a person is present. (Simulating an automated image stream from a Drone.)")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    # The image size (256, 256) and normalisation (dividing by 255) matching training pipeline.
    image_to_predict = image.resize((256, 256))
    image_to_predict = np.array(image_to_predict) / 255.0
    image_to_predict = np.expand_dims(image_to_predict, axis=0) # Add batch dimension

    # Make a prediction
    prediction = model.predict(image_to_predict)
    is_person = prediction[0][0] > 0.5
    
    # Display the prediction
    if is_person:
        st.subheader("Prediction: Person detected!")
        st.write("Does this image actually contain a person?")
        if st.button("Yes, a person is present"):
            st.success("Your selection matches the model's prediction.")
        if st.button("No, there is no person"):
            st.error("This was a False Positive.")
    else:
        st.subheader("Prediction: No person detected.")
        st.write("Does this image actually contain a person?")
        if st.button("Yes, a person is present"):
            st.error("This was a False Negative.")
        if st.button("No, a person is not present"):
            st.success("Your selection matches the model's prediction.")
