import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess the data
data = pd.read_csv(r"C:\Users\liamh\OneDrive\Desktop\fuelbuddy  (Responses) - Form Responses 1.csv")
prices = data['Price'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(prices)

# Create sequences (e.g., use the last 28 days to predict the next 7 days)
sequence_length = 31
X, y = [], []
for i in range(sequence_length, len(normalized_data) - 7):
    X.append(normalized_data[i-sequence_length:i])
    y.append(normalized_data[i:i+7])

X, y = np.array(X), np.array(y)

# Build the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=7))  # Predicting for 7 days

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)

# Predict the next week's prices
last_sequence = normalized_data[-sequence_length:]
predicted_sequence = model.predict(np.array([last_sequence]))
predicted_prices = scaler.inverse_transform(predicted_sequence)

# Interactive plotting using Plotly
actual_prices = scaler.inverse_transform(last_sequence)

hover_template = 'Day: %{x}<br>Price: %{y:$}'

fig = go.Figure()
fig.add_trace(go.Bar(x=np.arange(31, 38), y=predicted_prices[0],
                    name='Predicted Next 7 Days'))

# The same layout as provided in the original code
fig.update_layout(
    title='Fuel Price Prediction',
    xaxis_title='Days',
    yaxis_title='Price',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black'),
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[{'paper_bgcolor': 'white', 
                           'plot_bgcolor': 'white', 
                           'font': dict(color='black') }],
                    label='Light Theme',
                    method='relayout'
                ),
                dict(
                    args=[{'paper_bgcolor': 'black', 
                           'plot_bgcolor': 'black', 
                           'font': dict(color='white') }],
                    label='Dark Theme',
                    method='relayout'
                )
            ]),
            direction='down',
            showactive=True,
            x=1.25,
            xanchor='right',
            y=1.15,
            yanchor='top'
        )
    ]
)

# Display the figure
fig.show()