from olist_sentiment_analysis.dataloader.dataloader import DataLoader
import tensorflow as tf
import numpy as np

from utils.plot_image import display

from utils.config import Config

from configs.config import CFG






def predict(sentence: str) -> dict:
    
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('fashion_model_flask.h5')

    sequence = DataLoader.preprocess_data(sentence)

    predict_prob = model.predict(sequence)

    return {
        "text": sentence,
        "prob": predict_prob,
        "classe": np.where(predict_prob > 0.5, 1, 0)
    }