from flask import Flask, render_template
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)
model = pickle.load(open('Model_AirBersih_.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float+features)]
    prediction = model.predict(feature)