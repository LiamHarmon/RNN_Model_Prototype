import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess the data
df = pd.read_csv('FuelMonthlySpending.csv')

# Normalize the data
scaler = MinMaxScaler()
df['Total'] = scaler.fit_transform(df['Total'].values.reshape(-1, 1))

# Prepare the dataset for the RNN
X_train, y_train = [], []
look_back = 12  # Number of past weeks to consider for prediction
for i in range(look_back, len(df)):
    X_train.append(df['Total'][i-look_back:i])
    y_train.append(df['Total'][i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predicting the next week's fuel spending
last_12_weeks = np.array([df['Total'][-look_back:]])
last_12_weeks = np.reshape(last_12_weeks, (last_12_weeks.shape[0], last_12_weeks.shape[1], 1))
next_week_pred = model.predict(last_12_weeks)

# Inverse transform if data was normalized
next_week_pred_actual = scaler.inverse_transform(next_week_pred)

# Calculating RMSE on the training data
train_pred = model.predict(X_train)
train_pred_actual = scaler.inverse_transform(train_pred)
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(train_actual, train_pred_actual))

print("Next Week's Predicted Spending: ", next_week_pred_actual[0][0])
print("Root Mean Squared Error on Training Data: ", rmse)
