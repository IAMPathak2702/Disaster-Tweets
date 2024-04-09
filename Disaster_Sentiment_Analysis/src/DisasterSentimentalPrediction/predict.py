import tensorflow as tf
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config
import pandas as pd
import numpy as np
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.processing.data_handeling import load_pipeline


classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES_TO_PREDICT])
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