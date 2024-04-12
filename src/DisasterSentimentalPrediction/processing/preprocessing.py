import pandas as pd
import tensorflow as tf
from DisasterSentimentalPrediction.config import config  

class ModelCallbacks: 
    @staticmethod
    def early_stopping():
        """
        Create an EarlyStopping callback.

        Returns:
            tf.keras.callbacks.EarlyStopping: EarlyStopping callback object.
        """
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
        return early_stopping_callback
