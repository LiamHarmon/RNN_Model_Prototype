import pandas as pd
import numpy as np

# Load data
data = pd.read_csv(r"C:\Users\liamh\OneDrive\Desktop\FuelPrices.csv")

# Let's assume the CSV file has a column called 'Amount' which represents the amount spent
amounts = data['Price'].values

from sklearn.preprocessing import MinMaxScaler

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_amounts = scaler.fit_transform(amounts.reshape(-1, 1))

# If 'amounts' has 50 values, weeks_past will be [1, 2, ..., 50]
weeks_past = list(range(1, len(amounts) + 1))

# The next 4 weeks after the past data
weeks_future = list(range(len(amounts) + 1, len(amounts) + 5))


# Create sequence data
X, y = [], []
seq_length = 4

for i in range(len(scaled_amounts) - seq_length):
    X.append(scaled_amounts[i:i+seq_length])
    y.append(scaled_amounts[i+seq_length])

X = np.array(X)
y = np.array(y)

# Split into training and test (if needed)
train_size = int(len(X) * 0.67)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(50, input_shape=(seq_length, 1), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=10)

last_values = scaled_amounts[-seq_length:]
predictions = []

for _ in range(4):
    predicted_value = model.predict(last_values.reshape(1, seq_length, 1))
    predictions.append(predicted_value[0])
    last_values = np.append(last_values[1:], predicted_value)

predicted_prices = scaler.inverse_transform(predictions)

import plotly.graph_objs as go

# Define a color palette
color_past = '#636EFA'   # Royal Blue
color_future = '#EF553B'  # Coral Red
color_fill = '#FFDDC1'   # Light Coral for the shaded prediction area

# Actual past data
trace_past = go.Scatter(
    x=weeks_past,
    y=amounts,
    mode='lines+markers',
    name='Past Spending',
    line=dict(color=color_past, width=2.5),
    marker=dict(size=8, opacity=0.8, color=color_past)
)

# Predicted data
trace_future = go.Scatter(
    x=weeks_future,
    y=predicted_prices.flatten(),
    mode='lines+markers',
    name='Predicted Spending',
    line=dict(color=color_future, width=2.5, dash='dot'),
    marker=dict(size=8, color=color_future, symbol='star'),
    fill='tonexty',
    fillcolor=color_fill
)

# Layout with annotations
layout = go.Layout(
    title=dict(text='Past and Predicted Fuel Spending', x=0.5, font=dict(size=24)),
    xaxis=dict(title='Week', showgrid=False),
    yaxis=dict(title='Amount Spent (£)', showgrid=True, gridcolor='rgba(230,230,230,0.5)'),
    hovermode='closest',
    plot_bgcolor='rgba(250,250,250,1)',  # Lighter gray background color
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    annotations=[
        dict(
            x=weeks_future[0],
            y=predicted_prices[0][0],
            xref='x',
            yref='y',
            text='First Predicted Week',
            showarrow=True,
            arrowhead=4,
            ax=0,
            ay=-40
        )
    ],
    autosize=True,
    margin=dict(b=40, t=40, a=100, r=100)
)

fig = go.Figure(data=[trace_past, trace_future], layout=layout)
fig.show()
