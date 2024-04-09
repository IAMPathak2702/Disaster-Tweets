# Disaster Tweets - NLP Project

This project aims to classify tweets as either related to real disasters or not using Natural Language Processing (NLP) techniques. The dataset used for training and evaluation contains tweets collected during natural disasters.


## Introduction

Twitter is an essential platform for real-time information sharing, including updates during natural disasters. However, amidst a flurry of tweets, it's crucial to identify which ones are reporting actual incidents. This project leverages NLP techniques to classify tweets into two categories: those related to real disasters and those that are not.

## Dataset

The dataset used in this project consists of tweets labeled as either disaster-related or not. It includes text data along with corresponding labels. The dataset is split into training, validation, and test sets for model training, tuning, and evaluation.

To tabulate the provided data, you can organize it into a structured table format with columns for "text" and "target". Here's how you can tabulate the data:

| text                                                             | target |
|------------------------------------------------------------------|--------|
| Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all | 1      |
| Forest fire near La Ronge Sask. Canada                            | 1      |
| All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected | 1      |
| "13,000 people receive #wildfires evacuation orders in California " | 1      |
| Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school | 1      |
| #RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires | 1      |
| "#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas" | 1      |
| I'm on top of the hill and I can see a fire in the woods...       | 1      |

Each row represents a tweet with its corresponding target value. In this dataset, the "target" column indicates whether the tweet is related to a disaster (target = 1) or not (target = 0).

## Dependencies

- Python (>=3.6)
- TensorFlow (>=2.0)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn




## Model Architecture

The model architecture consists of a pre-trained word embedding layer followed by one or more recurrent neural network (RNN) layers such as LSTM or GRU. These layers are followed by fully connected layers with dropout for classification.

## Results



| Date                     | Mean Absolute Error | Mean Squared Error | Root Mean Squared Error | Accuracy | Loss   | Validation Mean Absolute Error | Validation Mean Squared Error | Validation Root Mean Squared Error | Validation Accuracy | Validation Loss |
|--------------------------|---------------------|---------------------|-------------------------|----------|--------|--------------------------------|--------------------------------|------------------------------------|----------------------|-----------------|
| 2024-03-17 02:30:18      | 0.3799              | 0.1852              | 0.4304                  | 0.7170   | 0.5459 | 0.2864                         | 0.1459                         | 0.3819                             | 0.7913               | 0.4508          |
| 2024-03-17 02:45:18      | 0.1914              | 0.0919              | 0.3032                  | 0.8803   | 0.3066 | 0.2531                         | 0.1525                         | 0.3905                             | 0.7900               | 0.4843          |
| 2024-03-17 03:00:18      | 0.0996              | 0.0452              | 0.2127                  | 0.9464   | 0.1690 | 0.2510                         | 0.1629                         | 0.4036                             | 0.7782               | 0.5453          |




![airflow](https://raw.githubusercontent.com/IAMPathak2702/Disaster-Tweets---NLP-project/main/airflow/pictures/dag_details.png)
![airflow_screenshot](https://raw.githubusercontent.com/IAMPathak2702/Disaster-Tweets---NLP-project/main/airflow/pictures/dag_list.png)
![airflow](https://raw.githubusercontent.com/IAMPathak2702/Disaster-Tweets---NLP-project/main/airflow/pictures/graph_view.png)

