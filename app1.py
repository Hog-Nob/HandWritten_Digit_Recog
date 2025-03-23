#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) for prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Preprocess the image
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    st.write(f"**Predicted Digit: {digit}**")


# In[2]:


import cv2
print(cv2.__version__)  # This will print the installed version of OpenCV



# In[ ]:




