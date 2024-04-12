import tensorflow as tf
from keras import layers
from DisasterSentimentalPrediction.config import config
from DisasterSentimentalPrediction.processing.data_handeling import load_dataset


def create_model():
    
    sentences,l = load_dataset(filename=config.TRAIN_FILE)
    # Input layer for text input
    text_vector = tf.keras.layers.TextVectorization(
        pad_to_max_tokens=True,
        max_tokens=10000,
        output_sequence_length=15
    )

    embedding = layers.Embedding(input_dim  = 10000,
                             output_dim = 128,
                             ) 
    text_vector.adapt(sentences)
    
    inputs = layers.Input(shape=(1,), dtype="string")
    x = text_vector(inputs)
    x = embedding(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model")

    return model
