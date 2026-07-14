import streamlit as st
st.title("Cat vs Dog Image classification")

#Testing the models
import tensorflow
import keras
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as img

# creating a object
model = keras.models.load_model('cat_dog.keras')
path = st.file_uploader("Upload a Image")

if st.button("Predict"):
    # path = '/content/flower/val/rose/24841052213_90fc2b1046_c.jpg'
    # class_labels = {0:"Cat", 1:"Dog"}
    #path = '/content/drive/MyDrive/new_set_belt/new_seat_belt_REFINED_RAW_DATASET/test/positive/opencv_frame_28 (4).png'
    test_image = load_img(path,target_size = (64,64,3)) #224 224
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis = 0)
    result = model.predict(test_image)
    # result = np.argmax(result)
    # output = class_labels[result][0][0]
    output = result[0][0]
    st.write(output)
    if output <0.5:
        output = "Cat"
    else:
        output = "Dog"
    st.write(output)
    st.image(path)