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
from DisasterSentimentalPrediction.processing.preprocessing import ModelCallbacks, df_to_tfdataset

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO


def perform_training():
    # Load train and test datasets
    train_dataset = load_dataset(config.TRAIN_FILE)
    test_dataset = load_dataset(config.EVAL_FILE)

    # Drop unnecessary features (if any)
    train_dataset = df_to_tfdataset(dataframe=train_dataset)
    test_dataset = df_to_tfdataset(dataframe=test_dataset)

    # Create model
    model = create_model()

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    callbacks = ModelCallbacks()
    checkpoint_callback = callbacks.create_model_checkpoint(dataframe=config.SAVE_MODEL_PATH)
    early_stopping_callback = callbacks.early_stopping()

    # Train the model
    model.fit(train_dataset,
              epochs=10,
              verbose=1,
              callbacks=[checkpoint_callback, early_stopping_callback],
              validation_data=test_dataset)


if __name__ == '__main__':
    perform_training()
