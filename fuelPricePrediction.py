import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model

from fuelPriceTraining import prepare_data

# Load the pre-trained RNN model
model = load_model('Fuel_Price_Predictor.keras')  # Replace with the path to your trained model

# Load your new, unseen data
new_data = pd.read_csv(r"C:\Users\liamh\OneDrive\Desktop\FuelPrices.csv")  # Replace with the path to your new data file
new_prices = new_data['Price'].values.reshape(-1, 1)

# Normalize the new data using the same scaler used during training
scaler = MinMaxScaler()  # Use the same scaler as during training
new_prices_scaled = scaler.fit_transform(new_prices)

# Prepare the input data with the same time_steps used during training
time_steps = 10  # Should be the same as during training
X_new, _ = prepare_data(new_prices_scaled, time_steps)  # Use the same prepare_data function as in the training code

# Use the pre-trained model to make predictions
predicted_new_prices = model.predict(X_new)

# Invert the scaling to get actual fuel prices
predicted_new_prices = scaler.inverse_transform(predicted_new_prices)


accurate_predictions = predicted_new_prices[-len(new_prices):]
print(accurate_predictions)

px.line(accurate_predictions, title='Predicted Fuel Prices')
px.line(new_prices, title='Actual Fuel Prices')
px.line(accurate_predictions, title='Predicted Fuel Prices').show()


# You now have predictions for the new, unseen fuel price data
