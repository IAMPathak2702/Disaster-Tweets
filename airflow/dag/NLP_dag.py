import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import tensorflow as tf
import keras
from keras import layers, Model
from keras.layers import Embedding, Dense, TextVectorization
import os
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


def preprocess_data(data: str):
    df = pd.read_csv(data)
    df.drop(["id", "keyword", "location"], axis=1,inplace=True)
    df.dropna(axis=0)
    os.makedirs(os.path.join("storage", "csvdata"), exist_ok=True)
    df.to_csv(f"storage/csvdata/processed{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")



def train_model(data: str):
    df = pd.read_csv(data)
    train_df_shuffled = df.sample(frac=1, random_state=42)
    train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(
        train_df_shuffled['text'].to_numpy(),
        train_df_shuffled['target'].to_numpy(),
        test_size=0.1,
        random_state=42)
    txtvector = TextVectorization(max_tokens=10000)
    txtvector.adapt(train_sentences)

    embedding = Embedding(input_dim=10000,
                          output_dim=128,
                          )

    inputs = layers.Input(shape=(1,), dtype="string")
    x = txtvector(inputs)
    x = embedding(x)
    x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(64, activation="relu")(x)  # optional dense layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs, name="model_Conv1D")

    # Compile Conv1D model
    model.compile(loss="binary_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])

    hist = model.fit(x=train_sentences,
                     y=train_labels,
                     epochs=5,
                     validation_data=(validation_sentences, validation_labels))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dag_directory = "storage"  # Get the directory of the current DAG file
    model_folder = os.path.join(dag_directory, "models")
    os.makedirs(model_folder, exist_ok=True)
    model.save(f"{model_folder}/model{timestamp}.keras",overwrite=True)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('disaster_tweet_analysis',
          default_args=default_args,
          description='Train a TensorFlow model to perform sentiment analysis on Twitter data related to Diasater',
          schedule_interval='@daily',
          )

# Define tasks
preprocess = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    op_args=["storage/data/train.csv"],
    dag=dag,
)

train= PythonOperator(
    task_id='train',
    python_callable=train_model,
    op_args=["storage/csvdata/processed.csv"],
    dag=dag,
)


# Set task dependencies
preprocess >> train