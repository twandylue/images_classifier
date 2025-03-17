from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
import os, io
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load models
models = {}
models_dir = './models'
for model_file in os.listdir(models_dir):
    if model_file.endswith('.h5'):
        model_name = model_file.split('.')[0]
        models[model_name] = tf.keras.models.load_model(os.path.join(models_dir, model_file))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']
    if file:
        img = Image.open(io.BytesIO(file.read()))
        # TODO:
        model = models['mnist_model']
        
        # Preprocess the image
        img = img.convert('L').resize((28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Predict
        prediction = model.predict(img_array)

        # Return the prediction and the name of the result
        return f"The prediction of number is: {prediction.argmax()}"

@app.route('/models')
def get_models():
    return jsonify({'models': list(models.keys())})

if __name__ == '__main__':
    app.run(debug=True, port=5001)