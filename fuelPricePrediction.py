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

test_sequence = new_prices_scaled[-time_steps:]
test_sequence = test_sequence.reshape(1, time_steps, 1)

predicted_price_scaled = model.predict(test_sequence)
predicted_price = scaler.inverse_transform(predicted_price_scaled)
print(f"Predicted price for the next day: {predicted_price[0][0]}")

px.line(accurate_predictions, title='Predicted Fuel Prices')
px.line(new_prices, title='Actual Fuel Prices')
px.line(accurate_predictions, title='Predicted Fuel Prices')

import plotly.graph_objects as go

# Extracting the predicted price for the next day
next_day_price = predicted_price[0][0]

# Appending the next day's predicted price to your accurate_predictions list
accurate_predictions_with_next_day = accurate_predictions.copy()
accurate_predictions_with_next_day = np.append(accurate_predictions_with_next_day, next_day_price)

# Plotting
fig = go.Figure()

# Plotting the actual fuel prices
fig.add_trace(go.Scatter(x=list(range(len(new_prices))), y=new_prices.reshape(-1), mode='lines+markers', name='Actual Fuel Prices'))

# Plotting the predicted fuel prices (including next day)
fig.add_trace(go.Scatter(x=list(range(len(accurate_predictions_with_next_day))), y=accurate_predictions_with_next_day, mode='lines+markers', name='Predicted Fuel Prices'))

# Highlighting the next day's predicted price
fig.add_trace(go.Scatter(x=[len(new_prices)-1], y=[next_day_price], mode='markers', marker=dict(color='red', size=10), name='Next Day Predicted Price'))

fig.update_layout(title="Fuel Prices Prediction")
fig.show()



# You now have predictions for the new, unseen fuel price data
