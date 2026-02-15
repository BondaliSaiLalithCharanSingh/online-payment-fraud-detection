from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os

# Load model from the correct path
model_path = os.path.join(os.path.dirname(__file__), 'payments.pkl')
model = pickle.load(open(model_path, "rb"))

app = Flask(__name__)

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/home")
def about1():
    return render_template('home.html')

@app.route("/predict")
def home1():
    return render_template('predict.html')

@app.route("/pred", methods=['POST', 'GET'])
def predict():
    x = [[x for x in request.form.values()]]
    print(x)
    
    x = np.array(x)
    print(x.shape)
    
    print(x)
    pred = model.predict(x)
    print(pred[0])
    
    if pred[0] == 0:
        prediction_text = "Not Fraud"
    else:
        prediction_text = "Is Fraud"
    
    return render_template('submit.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
