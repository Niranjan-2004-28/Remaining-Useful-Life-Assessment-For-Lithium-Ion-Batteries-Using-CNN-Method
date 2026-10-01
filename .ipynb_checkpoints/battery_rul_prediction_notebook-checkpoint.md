# Battery Remaining Useful Life (RUL) Prediction

## Project Overview
This notebook demonstrates a comprehensive approach to predicting the Remaining Useful Life (RUL) of batteries using machine learning techniques. The project includes data exploration, preprocessing, feature engineering, model building, and evaluation.

## Import Required Libraries


```python
# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
plt.style.use('ggplot')

# For statistical analysis
from scipy import stats

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# Save and load models
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Display settings
pd.set_option('display.max_columns', None)
```

## Load and Explore the Data


```python
# Load data
print("Loading battery data...")
df = pd.read_csv('Battery_RUL.csv')

# Display basic information
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
df.head()
```


```python
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove any missing values if present
df = df.dropna()
print(f"\nDataset shape after removing missing values: {df.shape}")
```


```python
# Summary statistics
print("\nSummary statistics:")
df.describe().T
```

## Exploratory Data Analysis (EDA)

### Distribution of Target Variable


```python
# Plot distribution of RUL
plt.figure(figsize=(10, 6))
plt.hist(df['RUL'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Remaining Useful Life (RUL)')
plt.ylabel('Frequency')
plt.title('Distribution of Battery Remaining Useful Life', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axvline(df['RUL'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {df["RUL"].mean():.2f}')
plt.axvline(df['RUL'].median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {df["RUL"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# Check for skewness
skewness = df['RUL'].skew()
print(f"Skewness of RUL: {skewness:.4f}")
if abs(skewness) > 0.5:
    print("The distribution is moderately skewed.")
elif abs(skewness) > 1:
    print("The distribution is highly skewed.")
else:
    print("The distribution is approximately symmetric.")
```

### Feature Analysis


```python
# Extract features for analysis
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
           'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
           'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

# Create pairplot for visual correlation analysis
plt.figure(figsize=(12, 8))
sns.pairplot(df[features + ['RUL']], corner=True, diag_kind='kde', 
             plot_kws={'alpha': 0.6, 's': 15, 'edgecolor': 'k', 'linewidth': 0.5})
plt.suptitle('Pairwise Relationships Between Features', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
```


```python
# Correlation analysis
plt.figure(figsize=(14, 10))
correlation = df[features + ['RUL']].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, 
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Sort and print feature correlations with RUL
feature_correlations = correlation['RUL'].sort_values(ascending=False)
print("Feature correlations with RUL:")
print(feature_correlations)
```

### Feature Distributions and Outliers


```python
# Box plots to identify outliers
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
plt.suptitle('Box Plots of Features - Outlier Detection', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
```


```python
# Distribution of features with density plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
plt.suptitle('Feature Distributions', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Check for skewness in features
skewness = df[features].skew()
print("Skewness of features:")
print(skewness)
```

### Relationship Between Features and Target


```python
# Create scatter plots of each feature against RUL
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    plt.scatter(df[feature], df['RUL'], alpha=0.5, edgecolor='k', s=30)
    plt.title(f'{feature} vs RUL')
    plt.xlabel(feature)
    plt.ylabel('RUL')
    # Add regression line
    m, b = np.polyfit(df[feature], df['RUL'], 1)
    plt.plot(df[feature], m*df[feature] + b, color='red', linestyle='--')
    plt.tight_layout()
plt.suptitle('Relationship Between Features and RUL', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
```

## Feature Engineering and Selection


```python
# Feature importance using SelectKBest (f_regression)
X = df[features]
y = df['RUL']

# Perform feature selection
selector = SelectKBest(f_regression, k='all')
selector.fit(X, y)

# Get scores and p-values
scores = selector.scores_
pvalues = selector.pvalues_

# Display feature importance
feature_scores = pd.DataFrame({
    'Feature': features,
    'Score': scores,
    'P-value': pvalues
})

feature_scores = feature_scores.sort_values('Score', ascending=False)
print("Feature importance based on F-regression:")
print(feature_scores)

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores, palette='viridis')
plt.title('Feature Importance (F-statistic)', fontsize=14)
plt.tight_layout()
plt.show()
```


```python
# Principal Component Analysis (PCA)
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(X_scaled_pca)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, 
        alpha=0.7, align='center', label='Individual variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         where='mid', label='Cumulative variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.title('Explained Variance by Components', fontsize=14)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Print the number of components needed for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% variance: {n_components_95}")
```

## Data Preprocessing and Splitting


```python
# Split features and target
X = df[features]
y = df['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
```


```python
# Try different scaling methods to see which works best
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

# Create a dict to store scaled data
scaled_data = {}

for name, scaler in scalers.items():
    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store the scaled data
    scaled_data[name] = (X_train_scaled, X_test_scaled)
    
    # Show first few rows of scaled train data
    print(f"\n{name} - First 3 rows of scaled training data:")
    print(pd.DataFrame(X_train_scaled[:3], columns=features))

# Decide on the best scaler - Standard Scaler is commonly used for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nSelected StandardScaler for the final model.")
```

## Baseline Model - Random Forest


```python
# Create a simple Random Forest model as baseline
print("Training baseline Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)

print(f"\nRandom Forest Baseline Model Metrics:")
print(f"MAE: {mae_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")
print(f"MAPE: {mape_rf:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=14)
plt.tight_layout()
plt.show()
```

## CNN Model Development

### Reshape Data for CNN


```python
# Reshape data for CNN (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

print(f"Training data shape after reshaping: {X_train_reshaped.shape}")
print(f"Test data shape after reshaping: {X_test_reshaped.shape}")
```

### Build CNN Model


```python
# Build CNN model
print("Building CNN model...")
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

# Print model summary
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
```

### Train and Monitor the Model


```python
# Set up callbacks for better training
checkpoint_path = "model_checkpoints/best_model.h5"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

# Train model
print("\nTraining CNN model...")
history = model.fit(
    X_train_reshaped, y_train, 
    validation_split=0.2, 
    epochs=100, 
    batch_size=32, 
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)
```

### Evaluate Model Performance


```python
# Evaluate model
y_pred = model.predict(X_test_reshaped)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Store metrics
metrics = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'MAPE': mape
}

print(f"\nCNN Model Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.4f}")

# Compare with baseline model
print("\nPerformance Comparison:")
print(f"{'Metric':<10} {'Random Forest':<15} {'CNN':<15}")
print(f"{'-'*40}")
print(f"{'MAE':<10} {mae_rf:<15.4f} {mae:<15.4f}")
print(f"{'MSE':<10} {mse_rf:<15.4f} {mse:<15.4f}")
print(f"{'RMSE':<10} {rmse_rf:<15.4f} {rmse:<15.4f}")
print(f"{'R²':<10} {r2_rf:<15.4f} {r2:<15.4f}")
print(f"{'MAPE':<10} {mape_rf:<15.4f} {mape:<15.4f}")
```

### Visualize Training History


```python
# Plot training history
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training', fontsize=14)
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training', fontsize=14)
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Extract final epoch metrics
final_epoch = len(history.history['loss'])
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

print(f"\nTraining completed in {final_epoch} epochs")
print(f"Final training loss: {final_train_loss:.4f}")
print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Final training MAE: {final_train_mae:.4f}")
print(f"Final validation MAE: {final_val_mae:.4f}")
```

### Visualize Predictions


```python
# Plot predictions vs actual
plt.figure(figsize=(14, 10))

# Scatter plot
plt.subplot(2, 1, 1)
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=40)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs Predicted RUL', fontsize=14)
plt.grid(True, alpha=0.3)

# Add text for metrics
plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Residual plot
plt.subplot(2, 1, 2)
residuals = y_test - y_pred.flatten()
plt.scatter(y_pred.flatten(), residuals, alpha=0.6, edgecolor='k', s=40)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted RUL')
plt.ylabel('Residuals')
plt.title('Residual Plot', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check distribution of residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(residuals.mean(), color='red', linestyle='dashed', linewidth=1, 
            label=f'Mean: {residuals.mean():.2f}')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean of residuals: {residuals.mean():.4f}")
print(f"Standard deviation of residuals: {residuals.std():.4f}")
```

## Save Model and Artifacts


```python
# Save model
model.save('battery_rul_cnn.h5')
joblib.dump(scaler, 'scaler.save')

# Save metrics
with open('model_metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# Save the Random Forest model as well
joblib.dump(rf_model, 'battery_rul_rf.joblib')

print("Models and artifacts saved successfully.")
```

## Prediction Function


```python
def predict_rul(input_data, model_type='cnn'):
    """
    Predict RUL from input data
    
    Parameters:
    -----------
    input_data : list or numpy array
        Features in the same order as training data:
        [Cycle_Index, Discharge Time, Decrement 3.6-3.4V, Max. Voltage Dischar., 
         Min. Voltage Charg., Time at 4.15V, Time constant current, Charging time]
    model_type : str, optional
        'cnn' or 'rf', default is 'cnn'
        
    Returns:
    --------
    float
        Predicted RUL
    """
    # Ensure input is numpy array
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data).reshape(1, -1)
    
    # Load scaler
    loaded_scaler = joblib.load('scaler.save')
    
    # Preprocess input
    scaled_input = loaded_scaler.transform(input_data)
    
    # Make prediction based on model type
    if model_type.lower() == 'cnn':
        # Load CNN model
        loaded_model = tf.keras.models.load_model('battery_rul_cnn.h5')
        # Reshape for CNN
        reshaped_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))
        # Predict
        prediction = loaded_model.predict(reshaped_input)
        return prediction[0][0]
    elif model_type.lower() == 'rf':
        # Load RF model
        loaded_model = joblib.load('battery_rul_rf.joblib')
        # Predict
        prediction = loaded_model.predict(scaled_input)
        return prediction[0]
    else:
        raise ValueError("Model type must be 'cnn' or 'rf'")
```

## Example Usage


```python
# Example usage
print("Testing prediction function with sample data...")

# Sample data: [Cycle_Index, Discharge Time, Decrement 3.6-3.4V, Max. Voltage Dischar., 
#               Min. Voltage Charg., Time at 4.15V, Time constant current, Charging time]
sample_data = [100, 120, 45, 3.9, 3.2, 200, 350, 500]

# Predict using both models
cnn_prediction = predict_rul(sample_data, model_type='cnn')
rf_prediction = predict_rul(sample_data, model_type='rf')

print(f"Sample input data: {sample_data}")
print(f"CNN model prediction: {cnn_prediction:.2f} cycles of RUL")
print(f"Random Forest prediction: {rf_prediction:.2f} cycles of RUL")
```

## Conclusions and Next Steps


```python
# Summary of findings
print("Project Summary and Conclusions:")
print("-" * 50)
print("1. Data Exploration:")
print("   - Analyzed battery health data with focus on Remaining Useful Life (RUL)")
print("   - Identified key features and their relationships with RUL")
print("   - Discovered correlations and distributions that inform model development")
print("\n2. Model Performance:")
print(f"   - CNN Model achieved R² of {r2:.4f} and RMSE of {rmse:.4f}")
print(f"   - Random Forest baseline achieved R² of {r2_rf:.4f} and RMSE of {rmse_rf:.4f}")
print(f"   - CNN {'outperformed' if r2 > r2_rf else 'underperformed compared to'} Random Forest")
print("\n3. Key Features:")
top_features = feature_importance['Feature'].iloc[:3].tolist()
print(f"   - Most important features from Random Forest: {', '.join(top_features)}")
print("\n4. Next Steps:")
print("   - Consider hyperparameter tuning for further optimization")
print("   - Explore other architectures (LSTM, GRU) for time series aspects")
print("   - Implement cross-validation for more robust evaluation")
print("   - Develop a feature selection pipeline to reduce dimensionality")
print("   - Create a web interface or API for deployment")
```

This comprehensive notebook provides a complete analysis of battery RUL prediction, from data exploration to model deployment. The CNN model demonstrates how deep learning can extract patterns from battery cycling data to predict remaining useful life.

