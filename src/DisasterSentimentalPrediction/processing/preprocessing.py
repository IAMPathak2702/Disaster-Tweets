import pandas as pd
import tensorflow as tf
from DisasterSentimentalPrediction.config import config  


def df_to_tfdataset(dataframe):
    """
    Perform feature engineering by dropping specified columns from the dataset.

    Returns:
        tuple: A tuple containing two pandas Series: (text, target).
    """
    # Load the data from CSV
    
    # Drop specified columns
    # _data = dataframe.drop(columns=config.COLUMNS_TO_DROP, axis=1)
    
    # Remove duplicate rows
    _data = dataframe.copy()
  
    
   
    # Separate target and text columns
    target = _data[config.TARGET]
    text = _data[config.FEATURES_TO_PREDICT]
    
    dataset = tf.data.Dataset.from_tensor_slices((text, target)).batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset
        


class ModelCallbacks:
    @staticmethod
    def create_model_checkpoint():
        """
        Create a ModelCheckpoint callback.

        Args:
            PACKAGE_ROOT (str): Root directory of the package.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback object.
        """

        # Define the file path to save the best model weights
        checkpoint_filepath = config.SAVE_MODEL_PATH

    # Define the ModelCheckpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=0
        )

        return model_checkpoint_callback
    
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
