from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
#from keras.preprocessing import image
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__,template_folder='template')

# Model saved with Keras model.save()
MODEL_PATH = 'Potholes_classification_final.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')



@app.route('/prediction', methods=["POST"])
def model_predict():
    img=image.load_img("img.jpg",target_size=(224,224))
    x=image.img_to_array(img) / 255
    x=np.expand_dims(img, axis=0)
    images=np.vstack([x])
    x=preprocess_input(x)
    prediction=model.predict(x)
    if prediction[0][0]<0.5:
         pred = 'Pothole'
    else:
         pred="No pothole"


    return render_template("prediction.html", data=pred)
    



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('html1.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
