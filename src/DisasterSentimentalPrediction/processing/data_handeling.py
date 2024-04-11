import os
import pandas as pd
import joblib
import tensorflow as tf 


from DisasterSentimentalPrediction.config import config

import pandas as pd

import pandas as pd

def load_dataset(filename):
    data = pd.read_csv(filename)
    label = data["target"]
    feature=data["text"]
    return (feature,label)




def save_pipeline(pipeline_to_save):
    try:
        save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
        tf.keras.models.save_model(pipeline_to_save)
        print(f"Model has been saved under the name :- {config.MODEL_NAME}")
    except Exception as e:
        print(f"Error saving pipeline from {save_path}: {e}")
        return None

# dataerialize the model
def load_pipeline(model_name):
    try: 
        save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
        model_loaded = tf.keras.models.load_model(save_path)
        print(f"Model has been loaded")
        return model_loaded
    except Exception as e:
        print(f"Error loading pipeline from {model_name}: {e}")
        return None