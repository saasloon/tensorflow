dataset_name = "daily-climate-time-series-data"
dataset_user = "sumanthvrao"

# Mount your Google Drive.
from google.colab import drive
drive.mount("/content/drive")

kaggle_creds_path = "./drive/MyDrive/TensorFlowSpecialization/kaggle.json"

! pip install kaggle --quiet

! mkdir ~/.kaggle
! cp $kaggle_creds_path ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

# ! kaggle competitions download -c {dataset_name}
! kaggle datasets download {dataset_user + "/" + dataset_name}

! mkdir kaggle_data
! unzip {dataset_name + ".zip"} -d kaggle_data

# Unmount your Google Drive
drive.flush_and_unmount()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Normalization
import matplotlib.pyplot as plt

# Define the path to your CSV file
data_path = "kaggle_data/DailyDelhiClimateTrain.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

# Parse the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Check if parsing was successful (no errors)
if df['date'].isnull().any():
    print("Warning: Some dates could not be parsed. Please investigate!")

# Select the window of previous data and the prediction days
prev_days = 30
prediction_days = 7
total_window = prev_days + prediction_days

# Define train and test split dates
train_end_date = df["date"].max() - pd.Timedelta(days=10*total_window)
train_data = df[df["date"] <= train_end_date]
test_data = df[df["date"] > train_end_date]

def data_transformation_layers_initialization(dataframe):
  # Extract the unique month string values from the month column
  vocabulary=df['date'].dt.month_name().unique()

  # Define the StringLookup layer
  string_lookup = keras.layers.StringLookup(vocabulary=vocabulary,
                                            output_mode='one_hot',
                                            num_oov_indices=0)
  # Define the Keras Normalization layer 
  normalizer = Normalization(axis=-1)

  # Extract the names of the numerical features for further normalization
  features = list(df.columns)[1:5]

  # Train the Normalization layer on the training data features
  normalizer.adapt(dataframe[features])

  return string_lookup, normalizer, features

string_lookup, normalizer, features = data_transformation_layers_initialization(df)

def data_transformation(dataframe, string_lookup, normalizer, features):
  # One hot month into a tensor
  month_encoded = string_lookup(dataframe['date'].dt.month_name())

  # Transform day of month into tensor
  day_of_month = tf.convert_to_tensor(dataframe["date"].dt.day.to_numpy())
  day_of_month = tf.expand_dims(day_of_month, axis=1)
  day_of_month = tf.cast(day_of_month, tf.float32)

  # Apply the trained Normalization layer to both train and test features
  scaled_numeric_features = normalizer(dataframe[features])

  # Create the final datasets with normalized features and one hot encoded features
  transformed_dataset = tf.concat([scaled_numeric_features,
                                   month_encoded,
                                   day_of_month],
                                  axis=1)
  
  return transformed_dataset

transformed_train_data = data_transformation(train_data, string_lookup, normalizer, features)
transformed_test_data = data_transformation(test_data, string_lookup, normalizer, features)
transformed_full_data = data_transformation(df, string_lookup, normalizer, features)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

sequence_length = total_window  # adjust based on your analysis of past data

train_sequences = create_sequences(transformed_train_data, sequence_length)
train_sequences = tf.random.shuffle(train_sequences, seed=213)

test_sequences = create_sequences(transformed_test_data, sequence_length)

full_sequences = create_sequences(transformed_full_data, sequence_length)

def get_features_and_labels(sequences, prev_days):
  sequences_features = sequences[:,:prev_days,:]
  sequences_labels = sequences[:,prev_days:,:]
  return sequences_features, sequences_labels

train_sequences_features, train_sequences_labels = get_features_and_labels(train_sequences, prev_days)
test_sequences_features, test_sequences_labels = get_features_and_labels(test_sequences, prev_days)
full_sequences_features, full_sequences_labels = get_features_and_labels(full_sequences, prev_days)

def create_model():
  input = tf.keras.Input(shape=(prev_days, train_sequences_features.shape[2]))
  x = LSTM(units=64, return_sequences=True)(input)
  x = LSTM(units=32)(x)
  output = Dense(units=prediction_days)(x)

  model = tf.keras.Model(inputs=input, outputs=output)
  model.compile(loss="mse", optimizer="adam")

  return model


models = []
for i in range(4):
  model = create_model()
  model.fit(train_sequences_features, train_sequences_labels[:,:,i],
          validation_data=(test_sequences_features, test_sequences_labels[:,:,i]),
          epochs=10, batch_size=32)
  models.append(model)

# Predict on test data
predicted_sequences = []
for i in range(4):
  predicted_ith_sequences = models[i].predict(full_sequences_features)
  predicted_sequences.append(predicted_ith_sequences)

def denormalize_data(data, normalizer, index, day=0):
  mean = normalizer.mean[0,index]
  variance = normalizer.variance[0,index]

  original_predictions = data * tf.sqrt(variance + 1e-7) + mean
  next_day_prediction = original_predictions[:,day]

  return next_day_prediction

day = 6
next_day_predictions = []
for i in range(4):
  next_day_prediction = denormalize_data(predicted_sequences[i], normalizer, i, day)
  next_day_predictions.append(next_day_prediction)

# Skip the first month based on Pandas indexing
filtered_df = df.iloc[30+day:30+day+next_day_predictions[0].shape[0]]

# Extract date and data for plotting
dates = filtered_df["date"]

def plot_graph(dates, real_data, data_column_name, prediction):
  data_column = real_data[data_column_name]

  plt.figure()

  # Plot Pandas data
  plt.plot(dates, data_column, label="Real data")

  # Plot NumPy array
  plt.plot(dates, prediction, label="Next day prediction")

  # Add labels, title, and legend
  plt.xlabel("Date")
  plt.ylabel("Value")
  plt.title("Comparison of values between real data and deep learning prediction")
  plt.legend()

  plt.show()


plot_graph(dates, filtered_df, "meantemp", next_day_predictions[0])
plot_graph(dates, filtered_df, "humidity", next_day_predictions[1])
plot_graph(dates, filtered_df, "wind_speed", next_day_predictions[2])
plot_graph(dates, filtered_df, "meanpressure", next_day_predictions[3])

