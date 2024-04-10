import tensorflow as tf
from keras import layers
from DisasterSentimentalPrediction.config import config
from DisasterSentimentalPrediction.processing.data_handeling import load_dataset
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import sys
sys.getdefaultencoding()

sentences = load_dataset(filename=config.TRAIN_FILE)
train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(
    sentences['text'].to_numpy(),
    sentences['target'].to_numpy(),
    test_size=0.1,
    random_state=42
)

def create_model():
    # Input layer for text input
    text_vector = tf.keras.layers.TextVectorization(
        pad_to_max_tokens=True,
        max_tokens=10000,
        output_sequence_length=15
    )

    text_vector.adapt(train_sentences)  # Pass train_sentences directly

    inputs = layers.Input(shape=(1,), dtype="string")
    x = text_vector(inputs)
    x = layers.Embedding(
        input_dim=10000,
        output_dim=128
    )(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    # x = layers.Dense(64, activation="relu")(x) # optional dense layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="model_Conv1D")

    return model
