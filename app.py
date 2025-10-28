import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
import cv2
from io import BytesIO

# Load the trained model
@st.cache_resource
def load_model():
    model = keras.models.load_model('mnist_model.h5')
    return model

model = load_model()

# Title
st.title("🔢 Handwritten Digit Classifier")
st.write("Draw a digit (0-9) and the AI will predict it!")

# Canvas for drawing
st.write("### Draw your digit:")
canvas_result = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if canvas_result is not None:
    # Load and preprocess image
    # image = Image.open(canvas_result).convert('L')  # Convert to grayscale
    # image = ImageOps.invert(image)  # Invert colors (white digit on black background)
    if canvas_result is not None:
    # Load and preprocess image
        image = Image.open(canvas_result).convert('L')
    
    # Smart inversion: only invert if background is mostly white
    img_array_check = np.array(image)
    if img_array_check.mean() > 127:
        image = ImageOps.invert(image)
    
    image = image.resize((28, 28))
    #image = image.resize((28, 28))  # Resize to 28x28
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image (processed)", width=200)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.reshape(1, 784) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    
    # Display results
    st.write("### Prediction Results:")
    st.success(f"**Predicted Digit: {predicted_digit}**")
    st.info(f"Confidence: {confidence:.2f}%")
    
    # Show all probabilities
    st.write("#### Probability for each digit:")
    for i in range(10):
        st.write(f"Digit {i}: {prediction[0][i]*100:.2f}%")