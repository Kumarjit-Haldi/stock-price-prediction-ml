# Stock Price Prediction Using Machine Learning
# Model: Linear Regression
# Dataset: Apple Inc. (AAPL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Download historical stock data
ticker = "AAPL"

data = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise ValueError("Stock data could not be downloaded.")

# 2. Get closing prices
prices = data["Close"].values.ravel()

# 3. Create features
# Previous 60 days -> next day's price
X = []
y = []

for i in range(60, len(prices)):
    X.append(prices[i - 60:i])
    y.append(prices[i])

X = np.array(X)
y = np.array(y)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\nModel Evaluation Results")
print("------------------------")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("MAPE :", round(mape, 2), "%")

# 8. Actual vs Predicted graph
plt.figure(figsize=(12, 6))

plt.plot(y_test, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")

plt.title("Actual vs Predicted AAPL Stock Price")
plt.xlabel("Test Data")
plt.ylabel("Stock Price (USD)")

plt.legend()
plt.grid(True)
plt.show()

# 9. Predict next day's price
last_60_days = prices[-60:].reshape(1, -1)

next_day_prediction = model.predict(last_60_days)[0]

print("\nPredicted Next-Day Closing Price: $",
      round(next_day_prediction, 2))
