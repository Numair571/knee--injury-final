from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model("keras_Modeld.h5", compile=False)

# Load the labels
class_names = open("labelsd.txt", "r").readlines()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def predict_class(data):
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

@app.route('/')
def home():
    return render_template('doubtful.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        data = preprocess_image(image_file)
        class_name, confidence_score = predict_class(data)
        return jsonify({'class_name': class_name[2:], 'confidence_score': float(confidence_score)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
