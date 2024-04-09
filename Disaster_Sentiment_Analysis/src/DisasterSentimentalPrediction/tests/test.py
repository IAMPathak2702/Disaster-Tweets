import tensorflow as tf
import pytest

def pytest_configure(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow.*")
    
import pytest
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.processing.data_handeling import load_dataset
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.config import config
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.processing.data_handeling import load_dataset
from Disaster_Sentiment_Analysis.src.DisasterSentimentalPrediction.predict import generate_predictions

# output from predict script not null
# output from predict script is str data type
# the output is Y for an example data

#Fixtures --> functions before test function --> ensure single_prediction

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction): # output is not none
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is string
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction): # check the output is Y
    assert single_prediction.get('prediction')[0] == 'Y'    