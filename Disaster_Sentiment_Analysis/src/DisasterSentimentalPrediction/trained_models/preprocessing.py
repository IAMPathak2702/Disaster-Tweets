import pandas as pd
import tensorflow as tf
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config  # Assuming config is a module containing COLUMNS_TO_DROP

class FeatureEngineering:
    """
    A class for performing feature engineering tasks.
    
    Attributes:
        columns (list): List of column names to drop from the dataset.
        data (str): Path to the input data file.
    """
    
    def __init__(self, data):
        """
        Initializes the FeatureEngineering class.

        Args:
            data (str): Path to the input data file.
        """
        self.columns = config.COLUMNS_TO_DROP
        self.data = data
        
    def drop(self):
        """
        Perform feature engineering by dropping specified columns from the dataset.

        Returns:
            tuple: A tuple containing two pandas Series: (text, target).
        """
        # Load the data from CSV
        _data = pd.read_csv(self.data)
        
        # Drop specified columns
        _data = _data.drop(self.columns, axis=1)
        
        # Remove duplicate rows
        _data = _data.drop_duplicates()
        
        # Separate target and text columns
        target = _data.pop("target")
        text = _data.pop("text")
        
        dataset = tf.data.Dataset.from_tensor_slices((text, target)).batch(32).prefetch(tf.data.AUTOTUNE)
    
        return dataset
        


class ModelCallbacks:
    @staticmethod
    def create_model_checkpoint(filepath):
        """
        Create a ModelCheckpoint callback.

        Args:
            PACKAGE_ROOT (str): Root directory of the package.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback object.
        """
        # Define the directory where you want to save the model
        

        # Create a timestamp string using the current datetime
        

        # Create the ModelCheckpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
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
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        return early_stopping_callback
