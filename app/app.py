import os
import requests
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify


from olist_sentiment_analysis.dataloader.dataloader import DataLoader

with open('fash.json', 'r') as f:
    model_json = f.read()



app = Flask(__name__)

@app.route('/predict/<string:sentence>', methods=['POST'])
def classify_sentence(sentence):

    sequence = DataLoader.preprocess_data(sentence)

    predict_prob = model.predict(sequence)

    return jsonify({
        "text": sentence,
        "prob": predict_prob,
        "classe": np.where(predict_prob > 0.5, 1, 0)
    })

app.run(port=5000, debug=False)