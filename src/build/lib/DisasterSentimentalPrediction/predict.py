import tensorflow as tf
from DisasterSentimentalPrediction.config import config
import pandas as pd
import numpy as np
from DisasterSentimentalPrediction.processing.data_handeling import load_pipeline , load_dataset


model = tf.keras.model.load_model(config.SAVE_MODEL_PATH)

def generate_predictions(data_input):
    data = data_input.copy()
    data = load_dataset(data)
    pred = model.predict(data)
    output = np.where(pred==1,'Disaster','Not A Disaster')
    result = {"prediction":output}
    return result

# def generate_predictions():
#     test_data = load_dataset(config.TEST_FILE)
#     pred = classification_pipeline.predict(test_data[config.FEATURES])
#     output = np.where(pred==1,'Y','N')
#     print(output)
#     #result = {"Predictions":output}
#     return output

if __name__=='__main__':
    generate_predictions()