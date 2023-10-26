import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv(r"C:\Users\liamh\OneDrive\Desktop\FuelPrices.csv")

# Assume 'Price' is the column with fuel prices
prices = data['Price'].values

# Normalize data (this helps in training)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Split data into sequences for training
sequence_length = 60  # Use past 60 days to predict the next

X, y = [], []

for i in range(sequence_length, len(prices_scaled)):
    X.append(prices_scaled[i-sequence_length:i, 0])
    y.append(prices_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()

# Add an RNN layer with 50 units
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(SimpleRNN(units=50))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=100, batch_size=32)

test_sequence = prices_scaled[-sequence_length:]
test_sequence = test_sequence.reshape(1, sequence_length, 1)

predicted_price_scaled = model.predict(test_sequence)
predicted_price = scaler.inverse_transform(predicted_price_scaled)
print(f"Predicted price for the next day: {predicted_price[0][0]}")

