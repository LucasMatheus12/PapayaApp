from flask import Flask, request
from tensorflow.keras.models import load_model
import pickle5 as pickle
import numpy as np
import json
import os

model_normal = load_model(r'model/papaya_classifier.h5')
model_gold = load_model(r'model/papaya_gold_classifier.h5')
CLASS_NAMES = ['MATURE', 'PARTIALLYMATURE', 'UNMATURE']

app = Flask(__name__)

def choose_model(type):
    return model_normal if type == 'normal' else model_gold

def preprocessing(img):
    img = np.array(img).reshape(-1, 180, 180, 3)
    return img


@app.route('/predict', methods=['POST'])
def serve_model():
    try:
        request_data = request.get_json(force=True)
        model = choose_model(request_data['type']) # Choose model
        img = preprocessing(request_data['img']) # Preprocessing image
        prediction = CLASS_NAMES[model.predict(img).argmax()] # Predict
        return json.dumps({'prediction': prediction})
    except:
        return json.dumps({'error': 'Error occur! Try Again!'})

@app.get('/')
def index():
    return 'App is runing!'

@app.get('/hello/<name>')
def hello(name):
    return f'Hello, {name}! {index()}'


if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )
