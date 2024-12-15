import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Load the trained model
model = tf.keras.models.load_model('fashion_mnist_classifier.h5')

# Class name for predictions
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Classifier")
st.text("Upload an image of clothing to classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file).convert("L").resize((28, 28))  # Convert to grayscale and resize
    image_array = np.array(image).astype("float32")  # Convert to float32
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = image_array.reshape(1, 28, 28)  # Reshape to match model input shape

    # Predict the class
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)


    # Display the results
    st.image(image, caption=f"Predicted: {predicted_class} (Confidence: {confidence:.2f})",  use_container_width=True)
    st.write(f"Prediction confidence: {confidence:.2f}")