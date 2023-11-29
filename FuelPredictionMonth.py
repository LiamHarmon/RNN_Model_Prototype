import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam

class FuelPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, file_path):
        # Load the CSV file
        data = pd.read_csv(file_path)
        # Assuming the file has a 'Price' column, we'll use it as the target variable
        price_data = data['Petrol'].values.reshape(-1, 1)
        # Normalize the price data
        normalized_data = self.scaler.fit_transform(price_data)
        # Convert the data into a format suitable for RNN (sequences)
        X, y = self.create_sequences(normalized_data)
        return np.array(X), np.array(y)

    def create_sequences(self, data, sequence_length=12):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(data[i + sequence_length])
        return X, y

    def create_rnn_model(self, input_shape, learning_rate=0.01):
        # Initialize the RNN model
        self.model = Sequential()
        # Add a SimpleRNN layer
        self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(SimpleRNN(units=50))
        # Add a Dense layer for output
        self.model.add(Dense(units=1))
        # Compile the model with a custom learning rate
        adam_optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

    def train_model(self, X, y, epochs=100, batch_size=32):
        # Train the model
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict_next_month(self, last_sequence):
        # Predict the next month's price using the last known data sequence
        last_sequence = last_sequence.reshape((1, last_sequence.shape[0], 1))
        predicted_price_normalized = self.model.predict(last_sequence)
        # Invert the normalization
        predicted_price = self.scaler.inverse_transform(predicted_price_normalized)
        return predicted_price[0, 0]

# Example usage:
predictor = FuelPricePredictor()
X, y = predictor.load_data(r"C:\Users\liamh\OneDrive\Desktop\Monthly_Fuel.csv")
predictor.create_rnn_model(input_shape=(X.shape[1], X.shape[2]),learning_rate=0.01)
predictor.train_model(X, y)
next_month_price = predictor.predict_next_month(X[-1])
print(f"Predicted price for next month: {next_month_price}")
