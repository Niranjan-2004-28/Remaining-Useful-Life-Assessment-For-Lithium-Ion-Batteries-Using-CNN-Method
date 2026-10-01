import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib


import tensorflow as tf
# For new prediction
def predict_new(input_data):
   # Load model and scaler
   loaded_model = tf.keras.models.load_model('battery_rul_cnn.h5')
   loaded_scaler = joblib.load('scaler.save')
   
   # Preprocess input
   scaled_input = loaded_scaler.transform([input_data])
   reshaped_input = scaled_input.reshape((1, scaled_input.shape[1], 1))
   
   # Predict
   prediction = loaded_model.predict(reshaped_input)
   return prediction[0][0]

# Example usage:
new_data = [3,7393.76,1112.992,4.249,3.224,5508.993,6762.02,10420.38]
predicted_rul = predict_new(new_data)

print(f"The RUL Predicted is  : {predicted_rul}")