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

# Load data
df = pd.read_csv('Battery_RUL.csv')
df = df.dropna()  # Remove any missing values

# Split features and target
X = df[['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
        'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
        'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']]
y = df['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for CNN (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_regularizer='l2'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train_reshaped, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stop])

# Evaluate model
y_pred = model.predict(X_test_reshaped)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2
}

# Save model and scaler
model.save('battery_rul_cnn.h5')
joblib.dump(scaler, 'scaler.save')

# Save metrics
with open('model_metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('training_history.png')
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs Predicted RUL')
plt.savefig('actual_vs_predicted.png')
plt.show()

print(f"Model Metrics:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.2f}")

##### For new prediction
####def predict_new(input_data):
####    # Load model and scaler
####    loaded_model = tf.keras.models.load_model('battery_rul_cnn.h5')
####    loaded_scaler = joblib.load('scaler.save')
####    
####    # Preprocess input
####    scaled_input = loaded_scaler.transform([input_data])
####    reshaped_input = scaled_input.reshape((1, scaled_input.shape[1], 1))
####    
####    # Predict
####    prediction = loaded_model.predict(reshaped_input)
####    return prediction[0][0]

# Example usage:
# new_data = [cycle_index, discharge_time, decrement_3.6-3.4V, max_voltage, min_voltage, time_4.15V, time_constant_current, charging_time]
# predicted_rul = predict_new(new_data)
