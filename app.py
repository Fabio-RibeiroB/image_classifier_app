from flask import Flask, render_template, request, redirect, flash, session
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from werkzeug.utils import secure_filename
import os
import logging
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.DEBUG)
logging.info('program starting')

model = load_model('models/imageclassifier.h5')

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.secret_key = b'=--9fh&&n?'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
        file = request.files['file'] 
        logging.debug(f'file ={file} ')

        if file.filename == '':
            flash('No selected file')
            return redirect('/')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(image_path)
            
            
            
            img = cv2.imread(image_path)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.savefig(UPLOAD_FOLDER + '/img.png')


            os.remove(image_path)

            resize = tf.image.resize(img, (256, 256))
            yhat = model.predict(np.expand_dims(resize/255, 0))
            if yhat > 0.5:
                result = f'Prediction: Sad. Confidence: {yhat[0][0]*100}%'
            else:
                result = f'Prediction: Happy. Confidence: {(1-yhat[0][0])*100}%'

            return render_template('after.html', data=result)



if __name__ == "__main__":
    app.run(debug=True)
