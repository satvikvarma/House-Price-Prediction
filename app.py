from flask import Flask, render_template, request 
import pandas as pd
import numpy as np
import pickle
import model 
from model import X

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
lr_clf = pickle.load(open('LinearRegression.pkl', 'rb'))

@app.route('/', methods = ['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    predicted_price = predict_price(location, sqft, bath, bhk)
    print(predicted_price)
    return render_template('index.html', predict=predicted_price)


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

