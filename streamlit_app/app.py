import streamlit as st
import numpy as np

import tensorflow as tf
from keras.models import load_model

from PIL import Image

import os

st.sidebar.title("About this Project")

justified_text = """
<div style="text-align: justify; line-height: 2.0;">
    This is a simple project to predict the images of certain products into their respective classes.
</div>

<div style="text-align: justify; line-height: 2.0;">
    The purpose of this project is for personal education and to understand how image classification works.Therefore, a simple CNN model is built instead of using any existing pre-trained models.
</div>

<div style="text-align: justify; line-height: 2.0;">
    The images are scraped and obtained from Google Images using an extension 'Download All Image'. The dataset is uploaded onto Kaggle.
</div>

<div style="text-align: justify; line-height: 2.0;">
    Further contributions are welcomed to improve the model in the GitHub link.
</div>
"""
st.sidebar.markdown(justified_text, unsafe_allow_html = True)

kaggle_link = "[Kaggle Link](https://www.kaggle.com/datasets/cliffordlee96/images-for-watches-shoes-headsets-laptops)"

st.sidebar.markdown(kaggle_link, unsafe_allow_html = True)

github_link = "[GitHub Link](https://github.com/clifford96/dl-image-class)"

st.sidebar.markdown(github_link, unsafe_allow_html = True)

model_file_path = os.path.join(os.path.dirname(__file__), 'model4.h5')

if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
    print("Model loaded successfully.")
else:
    print(f"Error: The file '{model_file_path}' does not exist.")

img_class = ['analogue',
             'digital',
             'smart',
             'headphone',
             'tws',
             'neckband',
             'gaming',
             'windows',
             'mac',
             '2in1',
             'loafers',
             'sneakers',
             'sports']

def preprocess_img(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis = 0) 
    return image

def predict(image):
    processed_image = preprocess_img(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = img_class[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_label, confidence

def predict_top_3(image, n = 3):
    processed_image = preprocess_img(image)
    predictions = model.predict(processed_image)[0]
    top_3_indices = np.argsort(predictions)[::-1][:n]
    top_3_labels = [img_class[i] for i in top_3_indices]
    top_3_confidences = [predictions[i] for i in top_3_indices]
    return top_3_labels, top_3_confidences

st.title("Image Classification Using CNN")

uploaded_image = st.file_uploader("Upload an image (Watch, Shoes, Laptops, Headsets)", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        top_3_labels, top_3_confidences = predict_top_3(image, n=3)
        st.write("Predictions:")
        for i in range(len(top_3_labels)):
            st.write(f"{top_3_labels[i]} - Confidence: {top_3_confidences[i]:.2f}")