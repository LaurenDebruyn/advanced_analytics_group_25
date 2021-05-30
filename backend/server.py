from typing import List, Tuple
from flask import Flask, render_template, url_for, request, jsonify, flash, redirect, Response
from PIL import Image
from tensorflow.keras import models
# from app import app
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix

LOAD_MODEL_1 = True
LOAD_MODEL_2 = False
LOAD_MODEL_3 = False
LOAD_MODEL_4 = True
LOAD_MODEL_5 = True
LOAD_MODEL_6 = False
LOAD_MODEL_7 = False


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

TOP_10_TAGS = ["tag_healthy", "tag_dinner", "tag_tasty", 
               "tag_vegan", "tag_breakfast", "tag_baking", 
               "tag_dessert", "tag_lunch", "tag_veggie", 
               "tag_cake"]

HEALTHY_TAG = ["tag_healthy"]

CAKE_TAG = ["tag_cake"]

MODEL_REPRESENTATIONS = [   
    ('model_1', 'Model 1', LOAD_MODEL_1, True),
    ('model_2', 'Model 2', LOAD_MODEL_2, False),
    ('model_3', 'Model 3 (broken)', LOAD_MODEL_3, False),
    ('model_4', 'Model 4', LOAD_MODEL_4, False),
    ('model_5', 'Model 5', LOAD_MODEL_5, False),
    ('model_6', 'Model 6 (is healthy)', LOAD_MODEL_6, False),
    ('model_7', 'Model 7 (is cake)', LOAD_MODEL_7, False)
]

PRECISSION = 0.5

default_chips = ['breakfast', 'avocado', 'brunch']
models = []

for model, _, enabled, _ in MODEL_REPRESENTATIONS:
    if enabled:
        models.append(keras.models.load_model(f'models/{model}'))
    else:
        models.append(None)


# model_1 = keras.models.load_model('models/model_1')
# model_2 = keras.models.load_model('models/model_2')
# model_4 = keras.models.load_model('models/model_4')
# model_5 = keras.models.load_model('models/model_5')
# model_6 = keras.models.load_model('models/model_6')
# model_7 = keras.models.load_model('models/model_7')


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', models=MODEL_REPRESENTATIONS, filename="placeholder.png")


@app.route('/image')
def show_image():
    return render_template('image.html')


@app.route('/', methods=["POST"])
def post_image():
    result = []
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        option = request.form['model']
        _, result = predict_tags(option, filename, TOP_10_TAGS)
        if not result:
            return redirect(request.url)
        return render_template('index.html', filename=filename, chips=result, models=MODEL_REPRESENTATIONS)
    else:
        return redirect(request.url)
        """
        elif 'model_4' in request.form:
            _, result = predict_tags(model_4, filename)
        elif 'model_5' in request.form:
            _, result = predict_tags(model_5, filename)
        if 'model_6' == option:
            _, result = predict_tags('model_6', filename, HEALTHY_TAG)
            if not result:
                result = ["not healthy"]
        elif 'model_7' == option:
            _, result = predict_tags('model_7', filename, CAKE_TAG)
            if not result:
                result = ["not cake"]
        """


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


def predict_tags(option: str, filename: str, labels: List[str]) -> Tuple[str, List[str]]:
    # get image
    img_path = f"static/uploads/{filename}"
    img = keras.preprocessing.image.load_img(img_path)
    img_array = keras.preprocessing.image.img_to_array(img)
    # resize image
    img_array_resized = cv2.resize(img_array, (150, 150))
    input_array = img_array_resized[np.newaxis,...]
    # predict tags
    if option == 'model_1':
        prediction = models[0].predict(input_array)[0]
    elif option == 'model_2':
        prediction = models[1].predict(input_array)[0]
    elif option == 'model_3':
        prediction = models[2].predict(input_array)[0]
    elif option == 'model_4':
        prediction = models[3].predict(input_array)[0]
    elif option == 'model_5':
        prediction = models[4].predict(input_array)[0]
    elif option == 'model_6':
        prediction = models[5].predict(input_array)[0]
    elif option == 'model_7':
        prediction = models[6].predict(input_array)[0]
    result = []
    for label, prob in zip(labels, prediction):
        if prob > PRECISSION:
            result.append(label)
    return prediction, result
