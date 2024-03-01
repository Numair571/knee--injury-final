import streamlit as st
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os

st.title('')

# Load your trained Keras model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def model_predict(img_path, model,class_names):
    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

img_file=st.file_uploader('Upload Image',type=['png','jpg','jpeg'])

def load_img(img):
    img=Image.open(img)
    return img

if img_file is not None:
    file_details={}
    file_details['name']=img_file.name
    file_details['size']=img_file.size
    file_details['type']=img_file.type
    st.write(file_details)
    st.image(load_img(img_file),width=255)

    with open(os.path.join('uploads','src.jpg'),'wb') as f:
        f.write(img_file.getbuffer())
    
    classname,_=model_predict('uploads/src.jpg',model,class_names)
    classname=classname[:-1]
    st.write(classname)