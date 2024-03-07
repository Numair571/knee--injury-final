from flask import Flask, render_template, request,redirect
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os
from pymongo import MongoClient

cluster=MongoClient('mongodb://127.0.0.1:27017')
db=cluster['knee']
users=db['users']

app = Flask(__name__)

# Load your trained Keras model
model = load_model("keras_model.h5", compile=False)
model1= load_model("keras_modeld.h5",compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
class_names1= open("labelsd.txt", "r").readlines()

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

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/main')
def mainPage():
    return render_template('main.html')

@app.route('/login',methods=['post','get'])
def login():
    user=request.form['username']
    password=request.form['password']
    res=users.find_one({"username":user})
    if res and dict(res)['password']==password:
        return render_template('main.html')
    else:
        return render_template('login.html',status='User does not exist or wrong password')


@app.route('/reg')
def reg():
    return render_template('signup.html')

@app.route('/regis',methods=['post','get'])
def register():
    username=request.form['username']
    password=request.form['password']
    k={}
    k['username']=username
    k['password']=password 
    res=users.find_one({"username":username})
    if res:
        return render_template('signup.html',status="Username already exists")
    else:
        users.insert_one(k)
        return render_template('signup.html',status="Registration successful")


def analyse(file_path,model,class_names):
    class_name, confidence_score = model_predict(file_path, model,class_names)
    print(class_name,len(class_name))
    if class_name.strip()=='Mild':
        return render_template('mild.html',status=class_name)
    elif class_name.strip()=='Moderate':
        return render_template('moderate.html',status=class_name)
    elif class_name.strip()=='Normal':
        return render_template('normal.html',status=class_name)
    elif class_name.strip()=='Severe':
        return render_template('severe.html',status=class_name)
    else:
        return render_template('doubtful.html',status=class_name)
@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Ensure the 'uploads' folder exists
        uploads_folder = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        # Save the file to ./uploads
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        class_name, confidence_score = model_predict(file_path, model,class_names)
        print(class_name,len(class_name))
        if class_name.strip()=='Mild':
            return render_template('mild.html',status=class_name)
        elif class_name.strip()=='Moderate':
            return render_template('moderate.html',status=class_name)
        elif class_name.strip()=='Normal':
            return render_template('normal.html',status=class_name)
        elif class_name.strip()=='Severe':
            return render_template('severe.html',status=class_name)
        else:
            return render_template('doubtful.html',status=class_name)
    return "invalid"

@app.route('/predict1', methods=['POST'])
def upload_file1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Ensure the 'uploads' folder exists
        uploads_folder = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        # Save the file to ./uploads
        file_path = os.path.join(uploads_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        class_name, confidence_score = model_predict(file_path, model1,class_names1)
        print(class_name,len(class_name))
        if class_name.strip()=='dummy_images':
            return render_template('main.html',status="Wrong Image Format")
        elif class_name.strip()=='fracture_images':
            class_name, confidence_score = model_predict(file_path, model,class_names)
            print(class_name,len(class_name))
            if class_name.strip()=='Mild':
                return render_template('mild.html',status=class_name)
            elif class_name.strip()=='Moderate':
                return render_template('moderate.html',status=class_name)
            elif class_name.strip()=='Normal':
                return render_template('normal.html',status=class_name)
            elif class_name.strip()=='Severe':
                return render_template('severe.html',status=class_name)
            else:
                return render_template('doubtful.html',status=class_name)
        

if __name__ == '__main__':
    app.run(port=5001, debug=True)