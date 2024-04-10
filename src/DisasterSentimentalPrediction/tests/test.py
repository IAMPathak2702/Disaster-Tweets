import warnings
import tensorflow as tf
import pytest
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.processing.data_handeling import load_dataset
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.predict import generate_predictions

# Filter DeprecationWarnings from TensorFlow
def pytest_configure(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow.*")

# Fixture to provide single prediction for testing
@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

# Test cases
def test_single_pred_not_none(single_prediction):
    # Check if prediction result is not None
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    # Check if prediction data type is string
    assert isinstance(single_prediction.get('prediction')[0], str)

def test_single_pred_validate(single_prediction):
    # Check if prediction matches the expected output ('Y' in this case)
    assert single_prediction.get('prediction')[0] == 'Disaster'
