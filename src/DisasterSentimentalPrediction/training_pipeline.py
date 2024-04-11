import sys
import os
sys.path.append("C:\\Users\\vpved\\Documents\\GitHub\\Disaster-Tweets\\src")


# Set TF_ENABLE_ONEDNN_OPTS environment variable to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import tensorflow as tf
from DisasterSentimentalPrediction.config import config
from DisasterSentimentalPrediction.processing.data_handeling import save_pipeline, load_dataset, load_pipeline
from DisasterSentimentalPrediction.pipeline import create_model
from DisasterSentimentalPrediction.processing.preprocessing import ModelCallbacks

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO


def perform_training():
    # Load train and test datasets
    train_feature , train_label = load_dataset(config.TRAIN_FILE)
    val_feature,val_label = load_dataset(config.VAL_FILE)

    # Create model
    model = create_model()

    # Compile model
    # Compile Conv1D model
    model.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])


    # Define callbacks
    callbacks = ModelCallbacks()
    checkpoint_callback = callbacks.create_model_checkpoint()
    early_stopping_callback = callbacks.early_stopping()

    # Train the model
    model.fit(x=train_feature,
              y=train_label,
              epochs=100,
              verbose=1,
              callbacks=[early_stopping_callback],
              validation_data=(val_feature,val_label)
              )
    
    model.save(config.SAVE_MODEL_PATH,save_format='tf')

if __name__ == '__main__':
    perform_training()
