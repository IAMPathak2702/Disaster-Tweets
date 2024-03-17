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
    df.drop(["id", "keyword", "location"], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)
    os.makedirs(os.path.join("storage", "csvdata"), exist_ok=True)
    processed_csv_path = f"storage/csvdata/processed_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(processed_csv_path, index=False)
    


def train_model():
    timestamp = datetime.now().strftime("%Y-%m-%d")
    dag_directory = "storage"  # Get the directory of the current DAG file
    model_folder = os.path.join(dag_directory, "models")
    os.makedirs(model_folder, exist_ok=True)

    df = pd.read_csv(f"storage/csvdata/processed_{datetime.now().strftime('%Y-%m-%d')}.csv")
    train_df_shuffled = df.sample(frac=1, random_state=42)
    train_sentences, validation_sentences, train_labels, validation_labels = train_test_split(
        train_df_shuffled['text'].to_numpy(),
        train_df_shuffled['target'].to_numpy(),
        test_size=0.1,
        random_state=42)
    
    train_labels = train_labels.astype('float32')
    validation_labels = validation_labels.astype('float32')


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
                  metrics=["accuracy", "MeanSquaredError", "MeanAbsoluteError","RootMeanSquaredError"])

    hist = model.fit(x=train_sentences,
                     y=train_labels,
                     epochs=100,
                     validation_data=(validation_sentences, validation_labels),
                     callbacks=[keras.callbacks.ModelCheckpoint(filepath=f"{model_folder}/model{timestamp}.keras",
                                                              save_best_only=True,
                                                              ),
                               keras.callbacks.EarlyStopping(patience=5)])
                     
    
    # Convert hist.history to a DataFrame
    hist_df = pd.DataFrame(hist.history)

    # Save hist.history to a CSV file
    hist_df.to_csv('storage/csvdata/results/tempresult/hist_history.csv', index=False)



def update_metrics_csv(csv_path, hist_csv_path):
    # Load hist_df from the CSV file
    hist_df = pd.read_csv(hist_csv_path)

    # Create DataFrame if it doesn't exist
    if not os.path.exists(csv_path):
        columns = ['datetime', 'loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error',
                   'root_mean_squared_error', 'val_loss', 'val_accuracy', 'val_mean_squared_error',
                   'val_mean_absolute_error', 'val_root_mean_squared_error']
        metrics_df = pd.DataFrame(columns=columns)
    else:
        # Load existing DataFrame
        metrics_df = pd.read_csv(csv_path)

    # Append recent values from hist_df
    recent_metrics = hist_df.iloc[-1].to_dict()
    metrics_df = pd.concat([metrics_df, pd.DataFrame([recent_metrics], columns=metrics_df.columns)], ignore_index=True)

    # Save DataFrame to CSV
    metrics_df.to_csv(csv_path, index=False)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('disaster_Tweet_Analysis',
          default_args=default_args,
          description='Train a TensorFlow model to perform sentiment analysis on Twitter data related to Disaster',
          schedule_interval='@daily',
          )

# Define tasks
preprocess = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    op_args=["storage/data/train.csv"],
    dag=dag,
)

train = PythonOperator(
    task_id='train',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

update = PythonOperator(
    task_id="update",
    python_callable=update_metrics_csv,
    op_args=['storage/csvdata/results/metrics_df.csv','storage/csvdata/results/tempresult/hist_history.csv'],
    dag=dag
)

# Set task dependencies
preprocess >> train >> update
