import logging
import tensorflow as tf
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.processing.data_handeling import save_pipeline,load_dataset,load_pipeline
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.pipeline import create_model
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.trained_models.preprocessing import FeatureEngineering , ModelCallbacks

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

def perform_training():
    # Load train and test datasets
    train_dataset = load_dataset(config.TRAIN_FILE)  
    test_dataset = load_dataset(config.TEST_FILE)  
    
    # Drop unnecessary features (if any)
    train_dataset = FeatureEngineering.drop(train_dataset)
    test_dataset = FeatureEngineering.drop(test_dataset)
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    callbacks = ModelCallbacks()
    checkpoint_callback = callbacks.create_model_checkpoint(filepath=config.SAVE_MODEL_PATH)
    early_stopping_callback = callbacks.early_stopping()
    
    # Train the model
    model.fit(train_dataset,
              epochs=10,
              verbose=1,
              callbacks=[checkpoint_callback, early_stopping_callback],
              validation_data=test_dataset,
              use_multiprocessing=True)
    
    
if __name__=='__main__':
    perform_training()
