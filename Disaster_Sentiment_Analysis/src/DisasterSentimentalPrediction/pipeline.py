import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.trained_models.preprocessing import FeatureEngineering 

#  Example of pretrained embedding with universal sentence encoder - https://tfhub.dev/google/universal-sentence-encoder/4
import tensorflow_hub as hub


def create_model():
    # Input layer for text input
    text_input = tf.keras.Input(shape=[], dtype=tf.string, name="text_input")

    # Universal Sentence Encoder layer
    sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                            trainable=False, name="USE")(text_input)

    # Dense layer with ReLU activation
    dense_1 = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(sentence_encoder_layer)

    # Output layer with sigmoid activation for binary classification
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(dense_1)

    # Construct the model
    model = tf.keras.Model(inputs=text_input, outputs=output, name="text_classification_model")
    
    return model
 