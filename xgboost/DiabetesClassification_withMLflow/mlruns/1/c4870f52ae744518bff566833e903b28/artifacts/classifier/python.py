#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained XGBoost model from the serialized file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json

    if data is None:
        return jsonify({'error': 'Invalid request data'})

    data_df = pd.DataFrame(data, index=[0])

    # Convert the data array to a DMatrix object
    dmatrix = xgb.DMatrix(data_df)

    # Make a prediction using the loaded XGBoost model
    prediction = model.predict(dmatrix)

    # Return the prediction as a JSON response
    response = {'prediction': int(prediction[0])}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[ ]:




