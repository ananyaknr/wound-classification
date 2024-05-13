import streamlit as st
import pathlib
from fastbook import *
from PIL import Image

st.title('Wound Classification')

temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

learn_inf = load_learner('upsampled-ENS-model.pkl')

def load_image(image_file):
    img = Image.open(image_file)
    return img

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = load_image(image_file)
    st.image(image, caption="Uploaded Image")
    pred_class, class_num, prob = learn_inf.predict(image)
    st.write("Prediction:", pred_class)
else:
    st.write("No image uploaded.")
