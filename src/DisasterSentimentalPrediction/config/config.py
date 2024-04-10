import pathlib
import os
import DisasterSentimentalPrediction

PACKAGE_ROOT = pathlib.Path(DisasterSentimentalPrediction.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT, "data")
TRAIN_FILE = os.path.join(DATAPATH, "train.csv")
TEST_FILE = os.path.join(DATAPATH, "test.csv")
EVAL_FILE = os.path.join(DATAPATH, "eval.csv")
MODEL_NAME = 'dsprediction.keras'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models', MODEL_NAME)


TARGET = 'target'

FEATURES = ["id","keyword","location","text"]

FEATURES_TO_PREDICT = "text"

STRING_FEATURES = ["test"]

COLUMNS_TO_DROP = ["id","keyword","location"]

